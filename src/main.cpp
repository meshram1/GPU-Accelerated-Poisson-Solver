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


int main(){
	constexpr int NX = 300;
	constexpr int NY = 300;
	constexpr double LX = 1.0;
	constexpr double LY = 1.0;
	double tol = 1e-5;
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
	double error = compute_error(A_new, Exact);
	int iter = 0;
	const int max_iter = 200000;
	
	double cpu_time = 0.0;
	double cpu_avg_time = 0.0;
	while (tol < error && iter < max_iter){
	        double start_time = get_time(); 
		boundary_conditions(A);
		solve_cpu(A,A_new,B,dx,dy);
		boundary_conditions(A_new);
		error = compute_error(A_new, Exact);		
		swap(A, A_new);
		double end_time = get_time();
		printf("iter %d error %.12e\n", iter, error);
		iter++;
		cpu_time += (end_time - start_time);
	}
	cpu_avg_time =  cpu_time/iter;
	
	
	// do allocation
	cublasHandle_t handle;
	
	CHECK_CUBLAS(cublasCreate(&handle));

	Tensor2D<double> A_gpu(NX,NY);
	Tensor2D<double> B_gpu(NX,NY);
	Tensor2D<double> An_gpu(NX,NY);
	Tensor2D<double> Exact_gpu(NX,NY);
	
	//fill in some data
	
	A_gpu.fill(0.1);
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
	
	double error_host = 1.0f;
	double* d_error = nullptr;
	
	cudaMalloc(&d_error, sizeof(double));
	cudaMemset(d_error, 0.0, sizeof(double));
	int N = NX*NY;
	double* d_diff = nullptr;
	//run the gpu kernel;
	double gpu_time = 0.0;
	double gpu_avg_time = 0.0;
	int iter_gpu = 0;
	double* d_A = A_gpu.device_data();
	double* d_An = An_gpu.device_data();
	double* d_Exact = Exact_gpu.device_data();
	cudaMalloc(&d_diff, N * sizeof(double)); 
	int n = NX*NY;
	while (tol < error_host && iter_gpu < max_iter){
	        double start_time_gpu = get_time(); 
	        
		bc_gpu<<<gridDim, blockDim>>>(d_A, NX, NY);
		solve_gpu<<<gridDim, blockDim>>>(d_A, d_An, B_gpu.device_data(), NX, NY, dx,dy);
		cudaDeviceSynchronize();
		bc_gpu<<<gridDim, blockDim>>>(d_An, NX, NY);
		
		//cudaMemset(d_error, 0, sizeof(double));
		//error_gpu<<<gridDim, blockDim>>>(An_gpu.device_data(), Exact_gpu.device_data(), d_error, NX, NY);
		CHECK_CUBLAS(cublasDcopy(handle, N, d_An, 1, d_diff, 1));
		const double minus_one = -1.0;
		CHECK_CUBLAS(cublasDaxpy(handle, N, &minus_one, d_Exact, 1, d_diff, 1));// basically -Exact + d_diff which is copy(d_An)

		// norm2 = ||d_diff||2
		double norm2 = 0.0;
		CHECK_CUBLAS(cublasDnrm2(handle, N, d_diff, 1, &norm2));

		// RMS
		error_host = norm2 / sqrtf((double)N);
		
		//error_finalize_gpu<<<1, 1>>>(d_error, NX, NY);
		//cudaMemcpy(&error_host, d_error, sizeof(double), cudaMemcpyDeviceToHost);


		//swap_arrays<<<(n + 255)/256, 256>>>(d_A, d_An, n);
		std::swap(d_A, d_An);
		
		double end_time_gpu = get_time();
		gpu_time += end_time_gpu - start_time_gpu;
		
		printf("iter %d error %.12e\n", iter_gpu, error_host);
		iter_gpu++;
		
	}
	gpu_avg_time =  gpu_time/iter_gpu;
	printf("For total iterations: %d, GPU error is %.12e\n", iter_gpu ,error_host);
	cout << "Average CPU time is " << cpu_avg_time*1e3 << " milliseconds!" << endl;
	cout << "Average GPU time is " << gpu_avg_time*1e3 << " milliseconds!" << endl;
	cout << "Total CPU time is " << cpu_time*1e3 << " milliseconds!" << endl;
	cout << "Total GPU time is " << gpu_time*1e3 << " milliseconds!" << endl;
	cudaFree(d_error);
	cudaFree(d_diff);
	cublasDestroy(handle);
}
