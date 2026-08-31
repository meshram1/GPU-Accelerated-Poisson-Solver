#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include "../include/poisson.cuh"
#include "../include/tensor.hpp"
#include <utility>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdlib>
using namespace std;
using namespace cfd;

#define BLOCK_SIZE 16

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}


// usage: ./run_gpu [N] [fixed_iters]
//   N            grid is N x N            (default 300)
//   fixed_iters  if > 0, both paths run exactly this many iterations and the
//                residual test is skipped. Use this to measure per-iteration
//                throughput at grid sizes where running to convergence would
//                take hours on the CPU.
int main(int argc, char** argv){
	const int N_ARG       = (argc > 1) ? atoi(argv[1]) : 300;
	const int FIXED_ITERS = (argc > 2) ? atoi(argv[2]) : 0;
	// third arg selects the solver: "jacobi" (default), "cg", or "both".
	// Positional and defaulted so sweep.sh's two-argument calls keep working.
	const char* SOLVER    = (argc > 3) ? argv[3] : "jacobi";
	const bool run_jacobi = (strcmp(SOLVER, "cg") != 0);
	const bool run_cg     = (strcmp(SOLVER, "jacobi") != 0);

	const int NX = N_ARG;
	const int NY = N_ARG;
	constexpr double LX = 1.0;
	constexpr double LY = 1.0;
	double tol = 1e-4;
	Tensor2D<double> A(NX,NY);
	Tensor2D<double> B(NX,NY); //source term
	Tensor2D<double> A_new(NX,NY);
	Tensor2D<double> Exact(NX,NY);
	
	double dx = LX/(NX-1);
	double dy = LY/(NY-1);
	//CPU STARTS
	A.fill(0.01);
	A_new.fill(0.0);
	Exact.fill(1.0);
	exact_solution(Exact,dx,dy);
	rhs(B, dx, dy);
		// RAND=1 replaces the single-eigenmode RHS with a random one.
	//
	// Why this is needed: CG terminates in at most (number of distinct
	// eigenvalues present in b) iterations. sin(pi x)sin(pi y) is an exact
	// EIGENVECTOR of the discrete Laplacian, so b spans a 1-dimensional Krylov
	// space and CG finishes in ONE step. Kernel fusion optimizes the per-
	// iteration inner loop, so a 1-iteration solve measures nothing but setup.
	// A random RHS has energy in every eigenmode -- which is also what a real
	// pressure-projection RHS looks like mid-simulation.
	//
	// L2-error-vs-exact is meaningless under RAND=1 (no analytic solution);
	// rel_res is the number that matters there.
	if (getenv("RAND")) {
		srand(12345);                       // fixed seed: runs are reproducible
		B.fill(0.0);
		for (std::size_t j = 1; j < B.ny() - 1; ++j)
			for (std::size_t i = 1; i < B.nx() - 1; ++i)
				B(i, j) = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
	}


	const double f_norm = rhs_norm_cpu(B);
	const int CHECK_EVERY = 100;

	double rel_res = 1.0;
	int iter = 0;
	const int max_iter = (FIXED_ITERS > 0) ? FIXED_ITERS : 200000;
	double cpu_start = get_time();
	while (run_jacobi && iter < max_iter) {
		boundary_conditions(A);
		solve_cpu(A, A_new, B, dx, dy);
		boundary_conditions(A_new);
		swap(A, A_new);
		++iter;

		if (FIXED_ITERS == 0 && iter % CHECK_EVERY == 0) {
			rel_res = residual_norm_cpu(A, B, dx, dy) / f_norm;
			printf("[cpu] iter %6d  rel_res %.6e\n", iter, rel_res);
			if (rel_res < tol) break;
		}
	}
	double cpu_time = get_time() - cpu_start;
	double cpu_avg_time = cpu_time / iter;
	double cpu_l2_error = compute_error(A, Exact);   // validation only
	
	
	// do allocation
	cublasHandle_t handle;
	
	CHECK_CUBLAS(cublasCreate(&handle));

	Tensor2D<double> A_gpu(NX,NY);
	Tensor2D<double> B_gpu(NX,NY);
	Tensor2D<double> An_gpu(NX,NY);
	Tensor2D<double> Exact_gpu(NX,NY);
	
	//fill in some data
	
	A_gpu.fill(0.01);   // must match the CPU initial guess for a fair comparison
	An_gpu.fill(1.0);
	Exact_gpu.fill(1.0);
	B_gpu.fill(1.0);
	
	//placing correct values;
	exact_solution_gpu(Exact_gpu,dx,dy);
	rhs_gpu(B_gpu,dx,dy);
	
	A_gpu.to_device();
	An_gpu.to_device();
	B_gpu.to_device();
	Exact_gpu.to_device();
	
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((NX+BLOCK_SIZE -1)/BLOCK_SIZE, (NY+BLOCK_SIZE -1)/BLOCK_SIZE);
	
	int N = NX*NY;
	//run the gpu kernel;
	int iter_gpu = 0;
	double* d_A = A_gpu.device_data();
	double* d_An = An_gpu.device_data();
	double* d_r = nullptr;
	cudaMalloc(&d_r, N * sizeof(double));
	cudaMemset(d_r, 0, N * sizeof(double));   // boundary entries stay 0 for the whole run

	const double f_norm_gpu = rhs_norm_cpu(B_gpu);  // host data of B_gpu is valid
	double rel_res_gpu = 1.0;
	cudaDeviceSynchronize();            // drain setup work before starting the clock
	double gpu_start = get_time();

	while (run_jacobi && iter_gpu < max_iter) {
		bc_gpu   <<<gridDim, blockDim>>>(d_A, NX, NY);
		solve_gpu<<<gridDim, blockDim>>>(d_A, d_An, B_gpu.device_data(), NX, NY, dx, dy);
		bc_gpu   <<<gridDim, blockDim>>>(d_An, NX, NY);
		std::swap(d_A, d_An);
		++iter_gpu;

		if (FIXED_ITERS == 0 && iter_gpu % CHECK_EVERY == 0) {
			residual_gpu<<<gridDim, blockDim>>>(d_A, B_gpu.device_data(), d_r, NX, NY, dx);
			double rnorm = 0.0;
			CHECK_CUBLAS(cublasDnrm2(handle, N, d_r, 1, &rnorm));  // implicit sync here
			rel_res_gpu = rnorm / f_norm_gpu;
			printf("[gpu] iter %6d  rel_res %.6e\n", iter_gpu, rel_res_gpu);
			if (rel_res_gpu < tol) break;
		}
	}

	cudaDeviceSynchronize();
	double gpu_time = get_time() - gpu_start;
	double gpu_avg_time = gpu_time / iter_gpu;

	Tensor2D<double> Result(NX, NY);
	cudaMemcpy(Result.host_data(), d_A, N * sizeof(double), cudaMemcpyDeviceToHost);
	double gpu_l2_error = compute_error(Result, Exact);
	if (run_jacobi){
	  printf("\n--- convergence ---\n");
	  printf("cpu: %6d iters, rel_res %.6e, L2 err vs exact %.6e\n", iter,     rel_res,      	cpu_l2_error);
	  printf("gpu: %6d iters, rel_res %.6e, L2 err vs exact %.6e\n", iter_gpu, rel_res_gpu, gpu_l2_error);
	  printf("\n--- timing (%dx%d) ---\n", NX, NY);
	  printf("cpu total %10.3f ms   avg/iter %8.5f ms\n", cpu_time*1e3, cpu_avg_time*1e3);
	  printf("gpu total %10.3f ms   avg/iter %8.5f ms\n", gpu_time*1e3, gpu_avg_time*1e3);
	  printf("speedup   %6.2fx total   %6.2fx per-iter\n",cpu_time/gpu_time, cpu_avg_time/gpu_avg_time) ;
	}
	
	
		// ---------------- Conjugate Gradient ----------------
	if (run_cg) {
		// b = -dx^2 * f, zero on the boundary.
		Tensor2D<double> b_cg(NX, NY);
		build_rhs_cg(b_cg, B, dx);

		const int cg_max_iter = (FIXED_ITERS > 0) ? FIXED_ITERS : 20000;

		// --- CPU ---
		Tensor2D<double> u_cpu(NX, NY);
		u_cpu.fill(0.0);                       // u0 = 0 satisfies the BC exactly
		CGResult cg_c = cg_cpu(u_cpu, b_cg, tol, cg_max_iter);
		double cg_cpu_l2 = compute_error(u_cpu, Exact);

		// --- GPU ---
		// Five device vectors. Every one starts fully zeroed so the boundary
		// entries are 0 and stay 0; only interior points are ever written.
		Tensor2D<double> u_gpu(NX, NY);
		u_gpu.fill(0.0);
		u_gpu.to_device();
		b_cg.to_device();

		double *d_rcg = nullptr, *d_pcg = nullptr, *d_Apcg = nullptr;
		cudaMalloc(&d_rcg,  N * sizeof(double));
		cudaMalloc(&d_pcg,  N * sizeof(double));
		cudaMalloc(&d_Apcg, N * sizeof(double));
		cudaMemset(d_rcg,  0, N * sizeof(double));
		cudaMemset(d_pcg,  0, N * sizeof(double));
		cudaMemset(d_Apcg, 0, N * sizeof(double));

		CGResult cg_g = cg_gpu(handle,
		                       u_gpu.device_data(), b_cg.device_data(),
		                       d_rcg, d_pcg, d_Apcg,
		                       NX, NY, tol, cg_max_iter);

		u_gpu.to_host();
		double cg_gpu_l2 = compute_error(u_gpu, Exact);

		printf("\n--- CG (%dx%d, tol %.1e) ---\n", NX, NY, tol);
		printf("cpu: %6d iters, rel_res %.6e, L2 err vs exact %.6e\n",
		       cg_c.iters, cg_c.rel_res, cg_cpu_l2);
		printf("gpu: %6d iters, rel_res %.6e, L2 err vs exact %.6e\n",
		       cg_g.iters, cg_g.rel_res, cg_gpu_l2);
		printf("cpu total %10.3f ms   avg/iter %8.5f ms\n",
		       cg_c.seconds * 1e3, cg_c.seconds * 1e3 / (cg_c.iters ? cg_c.iters : 1));
		printf("gpu total %10.3f ms   avg/iter %8.5f ms\n",
		       cg_g.seconds * 1e3, cg_g.seconds * 1e3 / (cg_g.iters ? cg_g.iters : 1));
		printf("speedup   %6.2fx total\n", cg_c.seconds / cg_g.seconds);
				// --- GPU, fused kernels ---
		// Same zero initial guess, same b, same tolerance: the only difference
		// from cg_g above is how the iteration is implemented.
		Tensor2D<double> u_gpuf(NX, NY);
		u_gpuf.fill(0.0);
		u_gpuf.to_device();
		cudaMemset(d_rcg,  0, N * sizeof(double));
		cudaMemset(d_pcg,  0, N * sizeof(double));
		cudaMemset(d_Apcg, 0, N * sizeof(double));

		CGResult cg_f = cg_gpu_fused(u_gpuf.device_data(), b_cg.device_data(),
		                             d_rcg, d_pcg, d_Apcg,
		                             NX, NY, tol, cg_max_iter, 10);
		u_gpuf.to_host();
		double cg_f_l2 = compute_error(u_gpuf, Exact);

		printf("\n--- CG: fused vs cuBLAS ---\n");
		printf("cublas: %6d iters, %9.3f ms, %8.5f ms/iter, rel_res %.6e, L2 %.6e\n",
		       cg_g.iters, cg_g.seconds*1e3,
		       cg_g.seconds*1e3/(cg_g.iters ? cg_g.iters : 1), cg_g.rel_res, cg_gpu_l2);
		printf("fused : %6d iters, %9.3f ms, %8.5f ms/iter, rel_res %.6e, L2 %.6e\n",
		       cg_f.iters, cg_f.seconds*1e3,
		       cg_f.seconds*1e3/(cg_f.iters ? cg_f.iters : 1), cg_f.rel_res, cg_f_l2);
		printf("fusion speedup %6.2fx\n", cg_g.seconds / cg_f.seconds);
		
		// --- GPU, mixed precision (unfused inner) ---
		// inner_tol must be <= the target tol, or the outer loop needs a second
		// refinement round. Each round RESTARTS CG and throws away the Krylov
		// space, which costs far more than the extra inner iterations save.
		const float INNER_TOL = 5e-5f;

		Tensor2D<double> u_mix(NX, NY);
		u_mix.fill(0.0);
		u_mix.to_device();

		CGResult cg_m = cg_gpu_mixed(handle, u_mix.device_data(), b_cg.device_data(),
		                             NX, NY, tol, /*max_outer=*/20,
		                             /*inner_iters=*/cg_max_iter,
		                             INNER_TOL, /*fused_inner=*/false);
		u_mix.to_host();
		double cg_m_l2 = compute_error(u_mix, Exact);

		// --- GPU, mixed precision + fused inner (the full stack) ---
		Tensor2D<double> u_fm(NX, NY);
		u_fm.fill(0.0);
		u_fm.to_device();

		CGResult cg_fm = cg_gpu_mixed(handle, u_fm.device_data(), b_cg.device_data(),
		                              NX, NY, tol, /*max_outer=*/20,
		                              /*inner_iters=*/cg_max_iter,
		                              INNER_TOL, /*fused_inner=*/true);
		u_fm.to_host();
		double cg_fm_l2 = compute_error(u_fm, Exact);

				// --- GPU, cuSPARSE CSR baseline ---
		Tensor2D<double> u_sp(NX, NY);
		u_sp.fill(0.0);
		u_sp.to_device();
		cudaMemset(d_rcg,  0, N * sizeof(double));
		cudaMemset(d_pcg,  0, N * sizeof(double));
		cudaMemset(d_Apcg, 0, N * sizeof(double));

		double sp_setup_ms = 0.0;
		CGResult cg_s = cg_gpu_cusparse(handle, u_sp.device_data(), b_cg.device_data(),
		                                d_rcg, d_pcg, d_Apcg,
		                                NX, NY, tol, cg_max_iter, &sp_setup_ms);
		u_sp.to_host();
		double cg_s_l2 = compute_error(u_sp, Exact);

		// ---- the ladder ----------------------------------------------------
		// Every row changes exactly ONE thing from the row above, so the
		// speedups are attributable rather than a single conflated number.
		// cuSPARSE is the only external baseline here; everything else is ours.
		const double base = cg_s.seconds;          // cuSPARSE CSR, FP64

		printf("\n================ CG ladder (%dx%d, tol %.0e) ================\n",
		       NX, NY, tol);
		printf("%-34s %7s %11s %11s %8s\n",
		       "config", "iters", "ms", "ms/iter", "vs base");
		printf("%-34s %7s %11s %11s %8s\n",
		       "----------------------------------", "-------",
		       "-----------", "-----------", "--------");

		#define ROW(label, res)                                                \
		    printf("%-34s %7d %11.1f %11.5f %7.2fx\n", label, (res).iters,     \
		           (res).seconds*1e3,                                          \
		           (res).seconds*1e3/((res).iters ? (res).iters : 1),          \
		           base/(res).seconds)

		ROW("cuSPARSE CSR + cuBLAS  fp64", cg_s);
		ROW("matrix-free  + cuBLAS  fp64", cg_g);
		ROW("matrix-free  + fused   fp64", cg_f);
		ROW("matrix-free  + cuBLAS  mixed", cg_m);
		ROW("matrix-free  + fused   mixed", cg_fm);
		#undef ROW

		printf("\ncuSPARSE also paid %.1f ms once to build and upload the matrix\n",
		       sp_setup_ms);
		printf("residuals: cusparse %.3e  free %.3e  fused %.3e  mixed %.3e  fused+mixed %.3e\n",
		       cg_s.rel_res, cg_g.rel_res, cg_f.rel_res, cg_m.rel_res, cg_fm.rel_res);
		printf("L2 vs sin*sin (meaningless under RAND=1): %.4e %.4e %.4e %.4e %.4e\n",
		       cg_s_l2, cg_gpu_l2, cg_f_l2, cg_m_l2, cg_fm_l2);
		printf("NOTE: mixed rows run FP32 inner iterations; cuSPARSE runs FP64.\n"
		       "      The FP64-vs-FP64 comparison is the defensible headline.\n");
		cudaFree(d_rcg);
		cudaFree(d_pcg);
		cudaFree(d_Apcg);
	}
	cudaFree(d_r);
	cublasDestroy(handle);
}
