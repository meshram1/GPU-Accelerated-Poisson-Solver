	#include<iostream>
#include<cmath>
#include "../include/tensor.hpp"
#include "../include/poisson.cuh"
#include <cuda_runtime.h>
#include <time.h>

using namespace std;
using namespace cfd;

void solve_cpu(Tensor2D<double>& A, Tensor2D<double>& A_new, Tensor2D<double>& f, double dx, double dy){
	double dx2 = dx*dx;
	double dy2 = dy*dy;
	for (std::size_t j = 1; j < A.ny()-1; ++j){
		for (std::size_t i = 1; i < A.nx()-1; ++i){
			A_new(i, j) = 0.25 * (A(i+1, j) + A(i-1, j) + A(i, j+1) + A(i, j-1) - dx2 * f(i, j));
		}
	}
}

__global__ void solve_gpu(const double* A, double* A_new, const double* f, int nx, int ny, double dx, double dy){
    double dx2 = dx*dx;
//    double dy2 = dy*dy; //same as dx^2
    int column = blockIdx.x * blockDim.x + threadIdx.x + 1; // 0, 1, 2, ..., nx-1, BC(0) and BC(nx-1)
    int  row = blockIdx.y * blockDim.y + threadIdx.y + 1; //
    if (column <= nx -2 && row <= ny - 2) {
    	int idx = column + row*nx;
        A_new[idx] = 0.25 * (A[idx + nx] + A[idx - nx] + A[idx + 1] + A[idx - 1] - dx2 * f[idx]);
  }
}

__global__ void bc_gpu(double* A, int nx, int ny){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 0, 1, 2, ..., nx-1, BC(0) and BC(nx-1);
    if (idx<nx) {
    	A[idx] = 0.0;
    	A[nx*(ny-1) + idx] = 0.0;
    }
    if (idx<ny){
    	A[idx*nx] = 0.0;
    	A[idx*nx + (nx-1)] = 0.0;
    }
  }

__global__ void error_gpu(double* A, double* A_exact, double* error, int nx, int ny){
   int column = blockIdx.x * blockDim.x + threadIdx.x; // 0, 1, 2, ..., nx-1, BC(0) and BC(nx-1);
   int  row = blockIdx.y * blockDim.y + threadIdx.y; 
   if (column < nx && row < ny){
   	int idx = column + row*nx;
   	double diff = A[idx] - A_exact[idx];
   	atomicAdd(error, diff*diff);
   }
}
// to prevent garbage data and race condition
__global__ void error_finalize_gpu(double* err_sum, int nx, int ny)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
       *err_sum = sqrt((*err_sum) / (double)(nx * ny));
           }
}

__global__ void swap_arrays(double* A, double* B, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
    }
}

//gpu based
void rhs_gpu(Tensor2D<double>& f, double dx, double dy){
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

//gpu based
void exact_solution_gpu(Tensor2D<double>& A, double dx, double dy){
   const double pi = M_PI;
   for (std::size_t j = 0; j < A.ny(); ++j){
   	double y = j*dy;
		for (std::size_t i = 0; i < A.nx(); ++i){
			double x = i*dx;			
			A(i, j) = std::sin(pi*x) * std::sin(pi*y);
		}
	}
}
