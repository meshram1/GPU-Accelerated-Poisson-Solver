#pragma once

#include<iostream>
#include<cmath>
#include "tensor.hpp" 
#include <cuda_runtime.h>
#include <time.h>
#include <utility>
#include <type_traits>
#include <cmath>
#include <cublas_v2.h>

using namespace std;
using namespace cfd;

extern int lx;
extern int ly;
extern int nx;
extern int ny;

inline void rhs(Tensor2D<double>& f, double dx, double dy){
   const double pi = M_PI;
   for (std::size_t j = 0; j < f.ny(); ++j){
   	double y = j*dy;
        for (std::size_t i = 0; i < f.nx(); ++i){
            double x = i*dx;
            f(i, j) = -2.0*pi*pi *
                      std::sin(pi*x) *
                      std::sin(pi*y);
		}
	}
}

inline void boundary_conditions(Tensor2D<double>& A){
	for (std::size_t j = 0; j < A.ny(); ++j){
		A(0, j) = 0.0;
		A(A.nx()-1, j) = 0.0;
	}
	for (std::size_t i = 0; i < A.nx(); ++i){
		A(i, 0) = 0.0;
		A(i, A.ny()-1) = 0.0;
	} 
}

inline void exact_solution(Tensor2D<double>& A, double dx, double dy){
   const double pi = M_PI;
   for (std::size_t j = 0; j < A.ny(); ++j){
   	double y = j*dy;
		for (std::size_t i = 0; i < A.nx(); ++i){
			double x = i*dx;
			A(i, j) = std::sin(pi*x) * std::sin(pi*y);
		}
	}
}

inline void initial_guess(Tensor2D<double>& A, double value){
	A.fill(value);
}

inline double compute_error(Tensor2D<double>& A, Tensor2D<double>& A_exact){
	double error = 0.0;
	for (std::size_t j = 0; j < A.ny(); ++j){
		for (std::size_t i = 0; i < A.nx(); ++i){
			error += std::pow(A(i, j) - A_exact(i, j), 2);
		}
	}
	return std::sqrt(error / (A.nx() * A.ny()));
}
// r = f - A*u, evaluated on interior points only.
// Boundary residual is identically zero because Dirichlet BCs are imposed exactly.
inline double residual_norm_cpu(Tensor2D<double>& A,
                                Tensor2D<double>& f,
                                double dx, double dy)
{
	const double inv_dx2 = 1.0 / (dx * dx);
	double sum = 0.0;
	for (std::size_t j = 1; j < A.ny() - 1; ++j) {
		for (std::size_t i = 1; i < A.nx() - 1; ++i) {
			double lap = (A(i+1, j) + A(i-1, j)
			            + A(i, j+1) + A(i, j-1)
			            - 4.0 * A(i, j)) * inv_dx2;
			double r = f(i, j) - lap;
			sum += r * r;
		}
	}
	return std::sqrt(sum);
}

// ||f|| over the same interior points, so the ratio is consistent.
inline double rhs_norm_cpu(Tensor2D<double>& f)
{
	double sum = 0.0;
	for (std::size_t j = 1; j < f.ny() - 1; ++j)
		for (std::size_t i = 1; i < f.nx() - 1; ++i)
			sum += f(i, j) * f(i, j);
	return std::sqrt(sum);
}

double get_time();


void solve_cpu(Tensor2D<double>& A,
               Tensor2D<double>& A_new,
               Tensor2D<double>& f,
               double dx, double dy);
               
__global__ void residual_gpu(const double* A, const double* f, double* r,
                             int nx, int ny, double dx);               
__global__ void solve_gpu(const double* A, double* A_new, const double* f, int nx, int ny, double dx, double dy);

__global__ void bc_gpu(double* A, int nx, int ny);

__global__ void error_gpu(double* A, double* A_exact, double* error, int nx, int ny);

__global__ void error_finalize_gpu(double* err_sum, int nx, int ny);

__global__ void swap_arrays(double* A, double* B, int n);

void rhs_gpu(Tensor2D<double>& f, double dx, double dy);

void exact_solution_gpu(Tensor2D<double>& A, double dx, double dy);

// ============================================================================
// Conjugate Gradient
//
// TL;DR: CG needs a symmetric POSITIVE definite matrix. The discrete Laplacian
// L is negative definite, so we solve the negated, dx^2-scaled system instead:
//
//     (A u)[i,j] = 4*u[i,j] - u[i+1,j] - u[i-1,j] - u[i,j+1] - u[i,j-1]
//         b[i,j] = -dx^2 * f[i,j]
//
// A = -dx^2 * L, so A is SPD and A u = b has the same solution as L u = f.
// Both scale factors cancel in the RELATIVE residual ||r||/||b||, which is why
// CG's reported convergence is directly comparable to Jacobi's.
//
// Boundary convention: every CG vector (u, r, p, Ap, b) holds exactly 0.0 on
// the boundary and is only ever written on the interior. Two consequences:
//   1. The stencil reading p[boundary]==0 IS the homogeneous Dirichlet BC.
//   2. Full-length dot/norm/axpy over all nx*ny entries are correct, because
//      the boundary zeros contribute nothing. No masking needed.
// ============================================================================

// Result of one solve, so CPU and GPU paths can be compared apples-to-apples.
struct CGResult {
    int    iters;      // iterations actually performed
    double rel_res;    // final ||b - A u|| / ||b||
    double seconds;    // wall time of the solve loop only (no setup)
};

// b = -dx^2 * f on the interior, 0.0 on the boundary.
// Built on the host once; the cost is trivial next to the solve.
inline void build_rhs_cg(Tensor2D<double>& b,
                         const Tensor2D<double>& f,
                         double dx)
{
    const double dx2 = dx * dx;
    b.fill(0.0);                                  // boundary starts and stays 0
    for (std::size_t j = 1; j < b.ny() - 1; ++j)
        for (std::size_t i = 1; i < b.nx() - 1; ++i)
            b(i, j) = -dx2 * f(i, j);
}

// Au = A*u on the interior. Does not touch the boundary, which stays 0.
inline void apply_A_cpu(const Tensor2D<double>& u, Tensor2D<double>& Au)
{
    for (std::size_t j = 1; j < u.ny() - 1; ++j)
        for (std::size_t i = 1; i < u.nx() - 1; ++i)
            Au(i, j) = 4.0 * u(i, j)
                     - u(i+1, j) - u(i-1, j)
                     - u(i, j+1) - u(i, j-1);
}

// Matrix-free CG on the host. u is both the initial guess and the output.
CGResult cg_cpu(Tensor2D<double>& u,
                const Tensor2D<double>& b,
                double tol, int max_iter);

// Same kernel as the CPU version above, one thread per interior point.
// This is CG's ONLY custom kernel -- everything else is cuBLAS.
__global__ void apply_A_gpu(const double* u, double* Au, int nx, int ny);

// Matrix-free CG on the device. d_u is the initial guess and the output.
// Caller owns all device buffers; this routine allocates nothing.
CGResult cg_gpu(cublasHandle_t handle,
                double* d_u, const double* d_b,
                double* d_r, double* d_p, double* d_Ap,
                int nx, int ny,
                double tol, int max_iter);

// Fused CG: same math as cg_gpu, but the scalars (alpha, beta, rsold, rsnew)
// stay resident in device memory and the vector operations merge into three
// kernels. check_every controls how often the host reads back the residual;
// CG converges in hundreds of iterations, not hundreds of thousands, so 10-25
// is right here (100 would overshoot badly).
CGResult cg_gpu_fused(double* d_u, const double* d_b,
                      double* d_r, double* d_p, double* d_Ap,
                      int nx, int ny, double tol, int max_iter,
                      int check_every);

// Mixed-precision CG via iterative refinement: FP64 residual and accumulation,
// FP32 inner solve. Allocates its own workspace. inner_tol should be LOOSE
// (1e-3 or so) -- driving the FP32 solve tighter than that wastes iterations
// on precision the outer loop is about to supply anyway.
CGResult cg_gpu_mixed(cublasHandle_t handle,
                      double* d_x, const double* d_b,
                      int nx, int ny,
                      double tol, int max_outer, int inner_iters,
                      float inner_tol, bool fused_inner);

// CG with cuSPARSE CSR SpMV in place of the matrix-free stencil. Same
// algorithm, same iteration count -- isolates the cost of a general sparse
// format. setup_ms receives the matrix build + upload time, which matrix-free
// does not pay (pass nullptr to ignore).
CGResult cg_gpu_cusparse(cublasHandle_t handle,
                         double* d_x, const double* d_b,
                         double* d_r, double* d_p, double* d_Ap,
                         int nx, int ny, double tol, int max_iter,
                         double* setup_ms);
