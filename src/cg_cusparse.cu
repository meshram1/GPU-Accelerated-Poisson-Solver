// ===========================================================================
// src/cg_cusparse.cu -- CG using cuSPARSE SpMV instead of the stencil kernel.
//
// TL;DR: this is the SAME CG algorithm as cg_gpu(). The only thing that
// changes is how A*p is computed:
//
//     matrix-free :  Ap[idx] = 4*p[idx] - p[idx±1] - p[idx±nx]
//                    coefficients are immediates, offsets are arithmetic
//
//     cuSPARSE    :  y[i] = sum_k values[k] * x[col_idx[k]]
//                    coefficients and column indices are READ FROM MEMORY
//
// Identical iteration counts, identical answer. Any wall-clock difference is
// attributable to exactly one thing: general sparse format vs. exploiting the
// known structure of the grid. That is the whole point of the comparison.
//
// The CSR matrix stores the same five numbers (4, -1, -1, -1, -1) once per
// row -- 21M nonzeros at 2048^2, about 270 MB -- and streams all of it through
// memory on EVERY iteration. The stencil stores nothing.
// ===========================================================================

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "../include/tensor.hpp"
#include "../include/poisson.cuh"

using namespace cfd;

#define CHK_SP(call) {                                                        \
    cusparseStatus_t s_ = (call);                                             \
    if (s_ != CUSPARSE_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "cuSPARSE error %s:%d: %d\n",                         \
                __FILE__, __LINE__, (int)s_);                                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

#define CHK_BL(call) {                                                        \
    cublasStatus_t s_ = (call);                                               \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                        \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                           \
                __FILE__, __LINE__, (int)s_);                                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

// ---------------------------------------------------------------------------
// Build the 5-point Laplacian in CSR on the host.
//
// The matrix is N x N with N = nx*ny -- the FULL grid, boundary included --
// so its vectors are layout-compatible with the matrix-free path and the two
// solvers can be compared directly.
//
// Boundary rows are identity. b is 0 on the boundary and x0 is 0 there, so
// those components stay 0 for the entire solve, which reproduces exactly what
// the matrix-free kernel does by never writing them.
// ---------------------------------------------------------------------------
static void build_csr_host(int nx, int ny,
                           std::vector<int>&    rowptr,
                           std::vector<int>&    colidx,
                           std::vector<double>& vals)
{
    const int N = nx * ny;
    rowptr.assign(N + 1, 0);
    colidx.clear();  colidx.reserve(5 * (size_t)N);
    vals.clear();    vals.reserve(5 * (size_t)N);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int idx = i + j * nx;
            rowptr[idx] = (int)vals.size();

            const bool interior = (i >= 1 && i <= nx - 2 &&
                                   j >= 1 && j <= ny - 2);
            if (!interior) {
                colidx.push_back(idx);
                vals.push_back(1.0);
                continue;
            }

            // CSR requires ASCENDING column indices within a row. For the
            // 5-point stencil that ordering is natural:
            //     idx-nx  <  idx-1  <  idx  <  idx+1  <  idx+nx
            colidx.push_back(idx - nx); vals.push_back(-1.0);
            colidx.push_back(idx - 1 ); vals.push_back(-1.0);
            colidx.push_back(idx     ); vals.push_back( 4.0);
            colidx.push_back(idx + 1 ); vals.push_back(-1.0);
            colidx.push_back(idx + nx); vals.push_back(-1.0);
        }
    }
    rowptr[N] = (int)vals.size();
}

CGResult cg_gpu_cusparse(cublasHandle_t handle,
                         double* d_x, const double* d_b,
                         double* d_r, double* d_p, double* d_Ap,
                         int nx, int ny, double tol, int max_iter,
                         double* setup_ms)
{
    const int N = nx * ny;

    // ---- SETUP: build and upload the matrix ------------------------------
    // Timed separately. Matrix-free pays none of this. In a time-stepping
    // solver it amortizes across timesteps; for a single solve it does not.
    const double t_setup0 = get_time();

    std::vector<int>    h_rowptr, h_colidx;
    std::vector<double> h_vals;
    build_csr_host(nx, ny, h_rowptr, h_colidx, h_vals);
    const int nnz = (int)h_vals.size();

    int    *d_rowptr = nullptr, *d_colidx = nullptr;
    double *d_vals   = nullptr;
    cudaMalloc(&d_rowptr, (N + 1)   * sizeof(int));
    cudaMalloc(&d_colidx, (size_t)nnz * sizeof(int));
    cudaMalloc(&d_vals,   (size_t)nnz * sizeof(double));
    cudaMemcpy(d_rowptr, h_rowptr.data(), (N + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, h_colidx.data(), (size_t)nnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals,   h_vals.data(),   (size_t)nnz * sizeof(double),
               cudaMemcpyHostToDevice);

    cusparseHandle_t sp = nullptr;
    CHK_SP(cusparseCreate(&sp));

    cusparseSpMatDescr_t matA = nullptr;
    CHK_SP(cusparseCreateCsr(&matA, N, N, nnz,
                             d_rowptr, d_colidx, d_vals,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Vector descriptors wrap fixed pointers. CG never swaps p or Ap, so
    // these can be created once outside the loop.
    cusparseDnVecDescr_t vecP = nullptr, vecAp = nullptr;
    CHK_SP(cusparseCreateDnVec(&vecP,  N, d_p,  CUDA_R_64F));
    CHK_SP(cusparseCreateDnVec(&vecAp, N, d_Ap, CUDA_R_64F));

    const double one = 1.0, zero = 0.0;
    size_t bufsize = 0;
    void*  dBuffer = nullptr;
    CHK_SP(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &one, matA, vecP, &zero, vecAp, CUDA_R_64F,
                                   CUSPARSE_SPMV_ALG_DEFAULT, &bufsize));
    if (bufsize > 0) cudaMalloc(&dBuffer, bufsize);

    cudaDeviceSynchronize();
    if (setup_ms) *setup_ms = (get_time() - t_setup0) * 1e3;

    printf("       [cusparse: %d nonzeros, %.1f MB CSR on device]\n",
           nnz, (nnz * (8.0 + 4.0) + (N + 1) * 4.0) / 1e6);

    // ---- SOLVE -----------------------------------------------------------
    // x0 = 0  =>  r0 = b - A*x0 = b,  p0 = r0.
    cudaMemset(d_x, 0, N * sizeof(double));
    CHK_BL(cublasDcopy(handle, N, d_b, 1, d_r, 1));
    CHK_BL(cublasDcopy(handle, N, d_b, 1, d_p, 1));

    double b_norm = 0.0, rsold = 0.0;
    CHK_BL(cublasDnrm2(handle, N, d_b, 1, &b_norm));
    CHK_BL(cublasDdot(handle, N, d_r, 1, d_r, 1, &rsold));

    CGResult out{0, 1.0, 0.0};
    if (b_norm == 0.0) { out.rel_res = 0.0; goto cleanup; }

    cudaDeviceSynchronize();
    {
    const double t0 = get_time();
    int k = 0;
    for (; k < max_iter; ++k) {
        // THE ONE LINE THAT DIFFERS FROM cg_gpu(): Ap = A*p via CSR SpMV
        // instead of a stencil kernel. Everything below is byte-identical.
        CHK_SP(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &one, matA, vecP, &zero, vecAp, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        double pAp = 0.0;
        CHK_BL(cublasDdot(handle, N, d_p, 1, d_Ap, 1, &pAp));
        if (pAp <= 0.0) break;

        const double alpha = rsold / pAp, nalpha = -alpha;
        CHK_BL(cublasDaxpy(handle, N, &alpha,  d_p,  1, d_x, 1));
        CHK_BL(cublasDaxpy(handle, N, &nalpha, d_Ap, 1, d_r, 1));

        double rsnew = 0.0;
        CHK_BL(cublasDdot(handle, N, d_r, 1, d_r, 1, &rsnew));
        if (std::sqrt(rsnew) / b_norm < tol) { ++k; rsold = rsnew; break; }

        const double beta = rsnew / rsold;
        CHK_BL(cublasDscal(handle, N, &beta, d_p, 1));
        CHK_BL(cublasDaxpy(handle, N, &one, d_r, 1, d_p, 1));
        rsold = rsnew;
    }
    cudaDeviceSynchronize();
    out.seconds = get_time() - t0;
    out.iters   = k;
    out.rel_res = std::sqrt(rsold) / b_norm;
    }

cleanup:
    if (dBuffer) cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecP);
    cusparseDestroyDnVec(vecAp);
    cusparseDestroy(sp);
    cudaFree(d_rowptr); cudaFree(d_colidx); cudaFree(d_vals);
    return out;
}
