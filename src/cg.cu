// ===========================================================================
// src/cg.cu -- matrix-free Conjugate Gradient, CPU and GPU.
//
// TL;DR: Jacobi needs O(N^2) iterations; CG needs O(N). At 300x300 that is the
// difference between ~166,600 sweeps and a few hundred. Each CG iteration is
// more expensive than a Jacobi sweep (one stencil + two dot products + three
// vector updates instead of one stencil), but nowhere near 100x more expensive,
// so CG wins by a wide margin overall.
//
// The algorithm, once per iteration:
//     Ap    = A*p
//     alpha = rsold / <p, Ap>
//     u    += alpha * p            <- step toward the solution
//     r    -= alpha * Ap           <- update residual without recomputing b - A*u
//     rsnew = <r, r>
//     beta  = rsnew / rsold
//     p     = r + beta * p         <- new search direction, A-conjugate to all
//                                     previous ones. That conjugacy is the whole
//                                     reason CG beats steepest descent.
//     rsold = rsnew
//
// On the GPU only apply_A_gpu is hand-written; the five vector operations are
// cuBLAS (Ddot, Daxpy, Dscal). Writing a reduction that hits peak bandwidth is
// a project in itself, and cuBLAS already did it.
// ===========================================================================

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/tensor.hpp"
#include "../include/poisson.cuh"

using namespace cfd;

#define CG_BLOCK 16

#define CG_CHECK_CUBLAS(call) {                                               \
    cublasStatus_t s = (call);                                                \
    if (s != CUBLAS_STATUS_SUCCESS) {                                         \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s);   \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

// ---------------------------------------------------------------------------
// Host helpers. These run over the FULL array, boundary included, which is safe
// because the boundary is identically zero in every CG vector -- see the note
// in poisson.cuh. Keeping them full-length matches what cuBLAS does on the GPU,
// so the two paths perform arithmetic in the same order where it matters.
// ---------------------------------------------------------------------------
static double dot_cpu(const Tensor2D<double>& x, const Tensor2D<double>& y)
{
    double s = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k)
        s += x.host_data()[k] * y.host_data()[k];
    return s;
}

// y = y + a*x
static void axpy_cpu(double a, const Tensor2D<double>& x, Tensor2D<double>& y)
{
    for (std::size_t k = 0; k < x.size(); ++k)
        y.host_data()[k] += a * x.host_data()[k];
}

// p = r + beta*p
static void xpby_cpu(const Tensor2D<double>& r, double beta, Tensor2D<double>& p)
{
    for (std::size_t k = 0; k < r.size(); ++k)
        p.host_data()[k] = r.host_data()[k] + beta * p.host_data()[k];
}

CGResult cg_cpu(Tensor2D<double>& u,
                const Tensor2D<double>& b,
                double tol, int max_iter)
{
    const std::size_t nx = u.nx(), ny = u.ny();

    Tensor2D<double> r(nx, ny), p(nx, ny), Ap(nx, ny);
    r.fill(0.0); p.fill(0.0); Ap.fill(0.0);   // boundaries pinned at 0 forever

    // r0 = b - A*u0. We take u0 = 0 (which satisfies the BC exactly), so this
    // collapses to r0 = b and we skip a stencil application.
    for (std::size_t k = 0; k < r.size(); ++k) {
        r.host_data()[k] = b.host_data()[k];
        p.host_data()[k] = b.host_data()[k];
    }

    const double b_norm = std::sqrt(dot_cpu(b, b));
    double rsold = dot_cpu(r, r);

    CGResult out{0, 1.0, 0.0};
    // Degenerate RHS: nothing to solve. Guard against dividing by zero below.
    if (b_norm == 0.0) { out.rel_res = 0.0; return out; }

    const double t0 = get_time();
    int k = 0;
    for (; k < max_iter; ++k) {
        apply_A_cpu(p, Ap);

        const double pAp = dot_cpu(p, Ap);
        // A is SPD, so pAp > 0 in exact arithmetic. A non-positive value means
        // round-off has broken down the recurrence; stop rather than produce
        // garbage.
        if (pAp <= 0.0) break;

        const double alpha = rsold / pAp;

        axpy_cpu( alpha, p,  u);   // u += alpha*p
        axpy_cpu(-alpha, Ap, r);   // r -= alpha*Ap

        const double rsnew = dot_cpu(r, r);

        // Converged? ||r||/||b||. Same metric the Jacobi path reports.
        if (std::sqrt(rsnew) / b_norm < tol) { ++k; rsold = rsnew; break; }

        xpby_cpu(r, rsnew / rsold, p);   // p = r + beta*p
        rsold = rsnew;
    }
    out.seconds = get_time() - t0;
    out.iters   = k;
    out.rel_res = std::sqrt(rsold) / b_norm;
    return out;
}

// ---------------------------------------------------------------------------
// GPU CG.
//
// Note on synchronization: cublasDdot with a HOST result pointer blocks until
// the reduction finishes, so there are two implicit syncs per iteration (one
// per dot product). That is unavoidable here -- alpha and beta are needed on
// the host to issue the next cuBLAS call. Eliminating it means keeping the
// scalars on the device (CUBLAS_POINTER_MODE_DEVICE) and computing alpha/beta
// in a tiny kernel, which is the natural next optimization. CG's iteration
// count advantage over Jacobi is so large that this is not the bottleneck yet.
// ---------------------------------------------------------------------------
CGResult cg_gpu(cublasHandle_t handle,
                double* d_u, const double* d_b,
                double* d_r, double* d_p, double* d_Ap,
                int nx, int ny,
                double tol, int max_iter)
{
    const int N = nx * ny;

    dim3 blockDim(CG_BLOCK, CG_BLOCK);
    dim3 gridDim((nx + CG_BLOCK - 1) / CG_BLOCK,
                 (ny + CG_BLOCK - 1) / CG_BLOCK);

    // u0 = 0, so r0 = b - A*u0 = b, and p0 = r0.
    CG_CHECK_CUBLAS(cublasDcopy(handle, N, d_b, 1, d_r, 1));
    CG_CHECK_CUBLAS(cublasDcopy(handle, N, d_b, 1, d_p, 1));

    double b_norm = 0.0;
    CG_CHECK_CUBLAS(cublasDnrm2(handle, N, d_b, 1, &b_norm));

    double rsold = 0.0;
    CG_CHECK_CUBLAS(cublasDdot(handle, N, d_r, 1, d_r, 1, &rsold));

    CGResult out{0, 1.0, 0.0};
    if (b_norm == 0.0) { out.rel_res = 0.0; return out; }

    cudaDeviceSynchronize();              // drain setup before starting the clock
    const double t0 = get_time();

    int k = 0;
    for (; k < max_iter; ++k) {
        apply_A_gpu<<<gridDim, blockDim>>>(d_p, d_Ap, nx, ny);

        double pAp = 0.0;
        CG_CHECK_CUBLAS(cublasDdot(handle, N, d_p, 1, d_Ap, 1, &pAp));
        if (pAp <= 0.0) break;            // see the CPU note on breakdown

        const double alpha  = rsold / pAp;
        const double nalpha = -alpha;

        CG_CHECK_CUBLAS(cublasDaxpy(handle, N, &alpha,  d_p,  1, d_u, 1));
        CG_CHECK_CUBLAS(cublasDaxpy(handle, N, &nalpha, d_Ap, 1, d_r, 1));

        double rsnew = 0.0;
        CG_CHECK_CUBLAS(cublasDdot(handle, N, d_r, 1, d_r, 1, &rsnew));

        if (std::sqrt(rsnew) / b_norm < tol) { ++k; rsold = rsnew; break; }

        // p = r + beta*p, expressed as scal-then-axpy because cuBLAS has no
        // single call for it.
        const double beta = rsnew / rsold;
        const double one  = 1.0;
        CG_CHECK_CUBLAS(cublasDscal(handle, N, &beta, d_p, 1));
        CG_CHECK_CUBLAS(cublasDaxpy(handle, N, &one, d_r, 1, d_p, 1));

        rsold = rsnew;
    }

    cudaDeviceSynchronize();
    out.seconds = get_time() - t0;
    out.iters   = k;
    out.rel_res = std::sqrt(rsold) / b_norm;
    return out;
}


// ===========================================================================
// FUSED CG
//
// TL;DR: identical math to cg_gpu above. Two things change:
//
//   1. MEMORY TRAFFIC. cublasDdot(p,Ap) re-reads two arrays that apply_A just
//      had in registers. A kernel boundary destroys registers, so the value
//      had to be spilled to VRAM and read back. Reducing INSIDE the stencil
//      kernel makes the dot product free. ~17N bytes/iter drops to ~12N.
//
//   2. SYNCHRONIZATION. cublasDdot with a HOST result pointer blocks until the
//      reduction lands -- twice per iteration -- because alpha and beta are
//      needed on the host to issue the next call. Keeping them in device
//      memory and computing them in a 1-thread kernel removes both stalls.
//
// Device scalar slots (d_s):
//     [0] rsold  [1] pAp  [2] alpha  [3] rsnew  [4] beta  [5] breakdown flag
// ===========================================================================

#define CGF_BLOCK 16
#define CGF_BS    (CGF_BLOCK * CGF_BLOCK)   // 256 threads per block

// Shared-memory tree reduction. Requires blockDim.x*blockDim.y == CGF_BS.
__device__ __forceinline__ double block_sum(double v, double* s)
{
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int stride = CGF_BS / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    return s[0];
}

// <x,x> over the interior -- used once at setup for ||b||^2.
__global__ void cgf_dot_self(const double* __restrict__ x,
                             double* __restrict__ partial, int nx, int ny)
{
    __shared__ double sh[CGF_BS];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    double v = 0.0;
    if (col <= nx - 2 && row <= ny - 2) v = x[col + row * nx];

    const double bs = block_sum(v * v, sh);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        partial[blockIdx.y * gridDim.x + blockIdx.x] = bs;
}

// FUSION 1: Ap = A*p AND <p,Ap>, one pass.
// pv and v are both live in registers when we multiply them, so the dot
// product costs zero extra memory traffic.
__global__ void cgf_Ap_dot(const double* __restrict__ p,
                           double* __restrict__ Ap,
                           double* __restrict__ partial, int nx, int ny)
{
    __shared__ double sh[CGF_BS];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    double pv = 0.0, v = 0.0;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        pv = p[idx];
        v  = 4.0 * pv - p[idx + 1] - p[idx - 1] - p[idx + nx] - p[idx - nx];
        Ap[idx] = v;
    }
    // Boundary threads contribute pv*v = 0*0, so no masking is needed.
    const double bs = block_sum(pv * v, sh);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        partial[blockIdx.y * gridDim.x + blockIdx.x] = bs;
}

// FUSION 2: u += alpha*p, r -= alpha*Ap, AND <r,r>.
// Was three cuBLAS calls that read r twice. alpha comes straight from device
// memory -- it was never on the host.
__global__ void cgf_update_dot(double* __restrict__ u,
                               double* __restrict__ r,
                               const double* __restrict__ p,
                               const double* __restrict__ Ap,
                               const double* __restrict__ d_s,
                               double* __restrict__ partial, int nx, int ny)
{
    __shared__ double sh[CGF_BS];
    const double alpha = d_s[2];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    double rv = 0.0;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        u[idx] += alpha * p[idx];
        rv = r[idx] - alpha * Ap[idx];
        r[idx] = rv;
    }
    const double bs = block_sum(rv * rv, sh);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        partial[blockIdx.y * gridDim.x + blockIdx.x] = bs;
}

// FUSION 3: p = r + beta*p.
// cuBLAS has no "xpby", so this was Dscal then Daxpy: two launches reading and
// writing p twice. Now one read of each, one write.
__global__ void cgf_p_update(double* __restrict__ p,
                             const double* __restrict__ r,
                             const double* __restrict__ d_s, int nx, int ny)
{
    const double beta = d_s[4];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        p[idx] = r[idx] + beta * p[idx];
    }
}

// Reduce per-block partials into one device slot. Launched <<<1,256>>>.
__global__ void cgf_reduce(const double* __restrict__ partial, int n,
                           double* __restrict__ d_s, int slot)
{
    __shared__ double sh[256];
    double v = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += partial[i];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_s[slot] = sh[0];
}

// Scalar updates, one thread each. These exist so alpha and beta never touch
// the host: a ~5us kernel launch beats a full device sync.
__global__ void cgf_alpha(double* d_s)
{
    const double pAp = d_s[1];
    // A is SPD so pAp > 0 exactly; <= 0 means round-off broke the recurrence.
    // Freeze the iteration and raise a flag the host reads on its next check.
    if (pAp <= 0.0) { d_s[2] = 0.0; d_s[5] = 1.0; }
    else            { d_s[2] = d_s[0] / pAp; }
}

__global__ void cgf_beta(double* d_s)
{
    const double rsold = d_s[0], rsnew = d_s[3];
    d_s[4] = (rsold > 0.0) ? rsnew / rsold : 0.0;
    d_s[0] = rsnew;                   // rotate for the next iteration
}

CGResult cg_gpu_fused(double* d_u, const double* d_b,
                      double* d_r, double* d_p, double* d_Ap,
                      int nx, int ny, double tol, int max_iter, int check_every)
{
    const int N = nx * ny;
    dim3 blockDim(CGF_BLOCK, CGF_BLOCK);
    dim3 gridDim((nx + CGF_BLOCK - 1) / CGF_BLOCK,
                 (ny + CGF_BLOCK - 1) / CGF_BLOCK);
    const int nblocks = gridDim.x * gridDim.y;

    double *d_partial = nullptr, *d_s = nullptr;
    cudaMalloc(&d_partial, nblocks * sizeof(double));
    cudaMalloc(&d_s, 6 * sizeof(double));
    cudaMemset(d_s, 0, 6 * sizeof(double));

    // u0 = 0  =>  r0 = b - A*u0 = b,  p0 = r0.
    cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice);

    cgf_dot_self<<<gridDim, blockDim>>>(d_b, d_partial, nx, ny);
    cgf_reduce<<<1, 256>>>(d_partial, nblocks, d_s, 0);      // rsold = <b,b>

    double h_s[6] = {0};
    cudaMemcpy(h_s, d_s, 6 * sizeof(double), cudaMemcpyDeviceToHost);
    const double b_norm = std::sqrt(h_s[0]);

    CGResult out{0, 1.0, 0.0};
    if (b_norm == 0.0) {
        out.rel_res = 0.0;
        cudaFree(d_partial); cudaFree(d_s);
        return out;
    }

    cudaDeviceSynchronize();
    const double t0 = get_time();

    int k = 0;
    for (; k < max_iter; ++k) {
        cgf_Ap_dot<<<gridDim, blockDim>>>(d_p, d_Ap, d_partial, nx, ny);
        cgf_reduce<<<1, 256>>>(d_partial, nblocks, d_s, 1);   // -> pAp
        cgf_alpha<<<1, 1>>>(d_s);                             // -> alpha

        cgf_update_dot<<<gridDim, blockDim>>>(d_u, d_r, d_p, d_Ap,
                                              d_s, d_partial, nx, ny);
        cgf_reduce<<<1, 256>>>(d_partial, nblocks, d_s, 3);   // -> rsnew
        cgf_beta<<<1, 1>>>(d_s);                              // -> beta, rotate

        // The ONLY host sync in the loop, once per check_every iterations.
        if ((k + 1) % check_every == 0) {
            cudaMemcpy(h_s, d_s, 6 * sizeof(double), cudaMemcpyDeviceToHost);
            if (h_s[5] != 0.0) { ++k; break; }                // breakdown
            if (std::sqrt(h_s[3]) / b_norm < tol) { ++k; break; }
        }

        cgf_p_update<<<gridDim, blockDim>>>(d_p, d_r, d_s, nx, ny);
    }

    cudaDeviceSynchronize();
    out.seconds = get_time() - t0;
    out.iters   = k;

    cudaMemcpy(h_s, d_s, 6 * sizeof(double), cudaMemcpyDeviceToHost);
    out.rel_res = std::sqrt(h_s[3]) / b_norm;

    cudaFree(d_partial);
    cudaFree(d_s);
    return out;
}

// ===========================================================================
// MIXED PRECISION CG (iterative refinement)
//
// TL;DR: on consumer Ada, FP64 runs at 1/64 the FP32 rate and moves twice the
// bytes. But FP32 alone stalls near rel_res 1e-7 -- it simply runs out of
// mantissa. Iterative refinement gets both: FP64 accuracy at FP32 speed.
//
//     x = 0                                (FP64)
//     repeat:
//         r = b - A*x                      (FP64)  <- accuracy lives here
//         if ||r||/||b|| < tol: done
//         solve A*d = r loosely in FP32    (FP32)  <- speed lives here
//         x += d                           (FP64)
//
// WHY IT WORKS: the SOLUTION needs FP64; the CORRECTION does not. A sloppy
// correction is just a slightly suboptimal step, and the next outer iteration
// cleans it up. A sloppy residual, by contrast, is permanent error.
//
// THE CRITICAL DETAIL IS SCALING. The residual shrinks by orders of magnitude
// each outer iteration. Cast it to FP32 raw and it underflows to zero after a
// few rounds. So we normalize r to unit norm before casting down, and scale
// the correction back up when accumulating. Skip this and refinement silently
// stops converging.
// ===========================================================================

// ---- FP32 stencil: the inner solve's only operator ------------------------
__global__ void cgm_Ap_f32(const float* __restrict__ p,
                           float* __restrict__ Ap, int nx, int ny)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        Ap[idx] = 4.0f * p[idx]
                - p[idx + 1]  - p[idx - 1]
                - p[idx + nx] - p[idx - nx];
    }
}

// ---- FP64 residual: r = b - A*x. The accuracy-critical step. --------------
__global__ void cgm_resid_f64(const double* __restrict__ x,
                              const double* __restrict__ b,
                              double* __restrict__ r, int nx, int ny)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        const double Ax = 4.0 * x[idx]
                        - x[idx + 1]  - x[idx - 1]
                        - x[idx + nx] - x[idx - nx];
        r[idx] = b[idx] - Ax;
    }
}

// ---- precision bridges ----------------------------------------------------
// f = (float)(d * scale). scale = 1/||r|| keeps the FP32 values near unit
// magnitude regardless of how small the residual has become.
__global__ void cgm_cast_down(const double* __restrict__ d,
                              float* __restrict__ f, double scale, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = (float)(d[i] * scale);
}

// x += scale * (double)f, undoing the normalization above.
__global__ void cgm_accum_up(double* __restrict__ x,
                             const float* __restrict__ f, double scale, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += scale * (double)f[i];
}

// ===========================================================================
// FP32 FUSED KERNELS -- the cgf_* treatment applied to the mixed-precision
// inner solve, so fusion and reduced precision compose instead of being
// mutually exclusive.
//
// These are float copies of cgf_Ap_dot / cgf_update_dot / cgf_p_update. The
// duplication is deliberate: templating the originals would touch code that
// is already measured and working, for no runtime benefit.
// ===========================================================================

__device__ __forceinline__ float block_sum_f32(float v, float* s)
{
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int stride = CGF_BS / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    return s[0];
}

// Ap = A*p AND <p,Ap>, one pass.
__global__ void cgf32_Ap_dot(const float* __restrict__ p,
                             float* __restrict__ Ap,
                             float* __restrict__ partial, int nx, int ny)
{
    __shared__ float sh[CGF_BS];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    float pv = 0.0f, v = 0.0f;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        pv = p[idx];
        v  = 4.0f * pv - p[idx + 1] - p[idx - 1] - p[idx + nx] - p[idx - nx];
        Ap[idx] = v;
    }
    const float bs = block_sum_f32(pv * v, sh);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        partial[blockIdx.y * gridDim.x + blockIdx.x] = bs;
}

// d += alpha*p ; r -= alpha*Ap ; <r,r>
__global__ void cgf32_update_dot(float* __restrict__ d,
                                 float* __restrict__ r,
                                 const float* __restrict__ p,
                                 const float* __restrict__ Ap,
                                 const float* __restrict__ d_s,
                                 float* __restrict__ partial, int nx, int ny)
{
    __shared__ float sh[CGF_BS];
    const float alpha = d_s[2];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

    float rv = 0.0f;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        d[idx] += alpha * p[idx];
        rv = r[idx] - alpha * Ap[idx];
        r[idx] = rv;
    }
    const float bs = block_sum_f32(rv * rv, sh);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        partial[blockIdx.y * gridDim.x + blockIdx.x] = bs;
}

// p = r + beta*p
__global__ void cgf32_p_update(float* __restrict__ p,
                               const float* __restrict__ r,
                               const float* __restrict__ d_s, int nx, int ny)
{
    const float beta = d_s[4];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (col <= nx - 2 && row <= ny - 2) {
        const int idx = col + row * nx;
        p[idx] = r[idx] + beta * p[idx];
    }
}

__global__ void cgf32_dot_self(const float* __restrict__ x,
                               float* __restrict__ partial, int nx, int ny)
{
    __shared__ float sh[CGF_BS];
    const int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    float v = 0.0f;
    if (col <= nx - 2 && row <= ny - 2) v = x[col + row * nx];
    const float bs = block_sum_f32(v * v, sh);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        partial[blockIdx.y * gridDim.x + blockIdx.x] = bs;
}

__global__ void cgf32_reduce(const float* __restrict__ partial, int n,
                             float* __restrict__ d_s, int slot)
{
    __shared__ float sh[256];
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += partial[i];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_s[slot] = sh[0];
}

__global__ void cgf32_alpha(float* d_s)
{
    const float pAp = d_s[1];
    if (pAp <= 0.0f) { d_s[2] = 0.0f; d_s[5] = 1.0f; }
    else             { d_s[2] = d_s[0] / pAp; }
}

__global__ void cgf32_beta(float* d_s)
{
    const float rsold = d_s[0], rsnew = d_s[3];
    d_s[4] = (rsold > 0.0f) ? rsnew / rsold : 0.0f;
    d_s[0] = rsnew;
}

// ---- inner FP32 CG, FUSED -------------------------------------------------
// Same algorithm as inner_cg_f32 below, but scalars stay on the device and the
// vector ops collapse into three kernels. Allocates its own scratch each call;
// the outer loop runs only a handful of times so that cost is negligible.
static int inner_cg_f32_fused(float* d, const float* rhs,
                              float* r, float* p, float* Ap,
                              int nx, int ny, float tol, int max_iter,
                              dim3 gridDim, dim3 blockDim, int check_every)
{
    const int N = nx * ny;
    const int nblocks = gridDim.x * gridDim.y;

    float *d_partial = nullptr, *d_s = nullptr;
    cudaMalloc(&d_partial, nblocks * sizeof(float));
    cudaMalloc(&d_s, 6 * sizeof(float));
    cudaMemset(d_s, 0, 6 * sizeof(float));

    cudaMemset(d, 0, N * sizeof(float));            // d0 = 0  =>  r0 = rhs
    cudaMemcpy(r, rhs, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(p, rhs, N * sizeof(float), cudaMemcpyDeviceToDevice);

    cgf32_dot_self<<<gridDim, blockDim>>>(rhs, d_partial, nx, ny);
    cgf32_reduce<<<1, 256>>>(d_partial, nblocks, d_s, 0);   // rsold = <rhs,rhs>

    float h_s[6] = {0};
    cudaMemcpy(h_s, d_s, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    const float rhs_norm = sqrtf(h_s[0]);
    if (rhs_norm == 0.0f) { cudaFree(d_partial); cudaFree(d_s); return 0; }

    int k = 0;
    for (; k < max_iter; ++k) {
        cgf32_Ap_dot<<<gridDim, blockDim>>>(p, Ap, d_partial, nx, ny);
        cgf32_reduce<<<1, 256>>>(d_partial, nblocks, d_s, 1);
        cgf32_alpha<<<1, 1>>>(d_s);

        cgf32_update_dot<<<gridDim, blockDim>>>(d, r, p, Ap, d_s,
                                                d_partial, nx, ny);
        cgf32_reduce<<<1, 256>>>(d_partial, nblocks, d_s, 3);
        cgf32_beta<<<1, 1>>>(d_s);

        if ((k + 1) % check_every == 0) {
            cudaMemcpy(h_s, d_s, 6 * sizeof(float), cudaMemcpyDeviceToHost);
            if (h_s[5] != 0.0f) { ++k; break; }
            if (sqrtf(h_s[3]) / rhs_norm < tol) { ++k; break; }
        }

        cgf32_p_update<<<gridDim, blockDim>>>(p, r, d_s, nx, ny);
    }

    cudaFree(d_partial);
    cudaFree(d_s);
    return k;
}

// ---- inner FP32 CG, plain cuBLAS ------------------------------------------
// Kept as the unfused reference so the two optimizations can be separated.
static int inner_cg_f32(cublasHandle_t h,
                        float* d, const float* rhs,
                        float* r, float* p, float* Ap,
                        int nx, int ny, float tol, int max_iter,
                        dim3 gridDim, dim3 blockDim)
{
    const int N = nx * ny;

    cudaMemset(d, 0, N * sizeof(float));           // d0 = 0  =>  r0 = rhs
    cublasScopy(h, N, rhs, 1, r, 1);
    cublasScopy(h, N, rhs, 1, p, 1);

    float rs = 0.0f, rhs_norm = 0.0f;
    cublasSdot(h, N, r, 1, r, 1, &rs);
    cublasSnrm2(h, N, rhs, 1, &rhs_norm);
    if (rhs_norm == 0.0f) return 0;

    int k = 0;
    for (; k < max_iter; ++k) {
        cgm_Ap_f32<<<gridDim, blockDim>>>(p, Ap, nx, ny);

        float pAp = 0.0f;
        cublasSdot(h, N, p, 1, Ap, 1, &pAp);
        if (pAp <= 0.0f) break;                    // FP32 breakdown

        const float a = rs / pAp, na = -a;
        cublasSaxpy(h, N, &a,  p,  1, d, 1);
        cublasSaxpy(h, N, &na, Ap, 1, r, 1);

        float rsnew = 0.0f;
        cublasSdot(h, N, r, 1, r, 1, &rsnew);
        if (sqrtf(rsnew) / rhs_norm < tol) { ++k; break; }

        const float beta = rsnew / rs, one = 1.0f;
        cublasSscal(h, N, &beta, p, 1);
        cublasSaxpy(h, N, &one, r, 1, p, 1);
        rs = rsnew;
    }
    return k;
}

// ---- outer driver ---------------------------------------------------------
CGResult cg_gpu_mixed(cublasHandle_t handle,
                      double* d_x, const double* d_b,
                      int nx, int ny,
                      double tol, int max_outer, int inner_iters,
                      float inner_tol, bool fused_inner)
{
    const int N = nx * ny;
    dim3 blockDim(CGF_BLOCK, CGF_BLOCK);
    dim3 gridDim((nx + CGF_BLOCK - 1) / CGF_BLOCK,
                 (ny + CGF_BLOCK - 1) / CGF_BLOCK);
    const int flat = (N + 255) / 256;

    double* d_r64 = nullptr;
    float  *f_rhs = nullptr, *f_d = nullptr, *f_r = nullptr,
           *f_p   = nullptr, *f_Ap = nullptr;
    cudaMalloc(&d_r64, N * sizeof(double));
    cudaMalloc(&f_rhs, N * sizeof(float));
    cudaMalloc(&f_d,   N * sizeof(float));
    cudaMalloc(&f_r,   N * sizeof(float));
    cudaMalloc(&f_p,   N * sizeof(float));
    cudaMalloc(&f_Ap,  N * sizeof(float));
    cudaMemset(d_r64, 0, N * sizeof(double));
    cudaMemset(f_rhs, 0, N * sizeof(float));
    cudaMemset(f_d,   0, N * sizeof(float));
    cudaMemset(f_r,   0, N * sizeof(float));
    cudaMemset(f_p,   0, N * sizeof(float));
    cudaMemset(f_Ap,  0, N * sizeof(float));

    cudaMemset(d_x, 0, N * sizeof(double));        // x0 = 0

    double b_norm = 0.0;
    cublasDnrm2(handle, N, d_b, 1, &b_norm);

    CGResult out{0, 1.0, 0.0};
    if (b_norm == 0.0) { out.rel_res = 0.0; goto cleanup; }

    cudaDeviceSynchronize();
    {
    const double t0 = get_time();
    int total_inner = 0, outer = 0;
    double rnorm = 0.0;

    for (; outer < max_outer; ++outer) {
        // --- FP64: the residual. Everything else may be sloppy; this may not.
        cgm_resid_f64<<<gridDim, blockDim>>>(d_x, d_b, d_r64, nx, ny);
        cublasDnrm2(handle, N, d_r64, 1, &rnorm);
        if (rnorm / b_norm < tol) break;

        // --- normalize, then drop to FP32 (see the scaling note above)
        const double scale = 1.0 / rnorm;
        cgm_cast_down<<<flat, 256>>>(d_r64, f_rhs, scale, N);

        // --- FP32: the expensive part, solved loosely
        total_inner += fused_inner
            ? inner_cg_f32_fused(f_d, f_rhs, f_r, f_p, f_Ap,
                                 nx, ny, inner_tol, inner_iters,
                                 gridDim, blockDim, /*check_every=*/10)
            : inner_cg_f32(handle, f_d, f_rhs, f_r, f_p, f_Ap,
                           nx, ny, inner_tol, inner_iters,
                           gridDim, blockDim);

        // --- FP64: accumulate the correction, undoing the normalization
        cgm_accum_up<<<flat, 256>>>(d_x, f_d, rnorm, N);
    }

    cudaDeviceSynchronize();
    out.seconds = get_time() - t0;
    out.iters   = total_inner;                     // FP32 iterations, the real work
    out.rel_res = rnorm / b_norm;
    printf("       [mixed%s: %d outer refinements, %d inner FP32 iters]\n",
           fused_inner ? "+fused" : "", outer, total_inner);
    }

cleanup:
    cudaFree(d_r64); cudaFree(f_rhs); cudaFree(f_d);
    cudaFree(f_r);   cudaFree(f_p);   cudaFree(f_Ap);
    return out;
}
