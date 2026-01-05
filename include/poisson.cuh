#pragma once

#include<iostream>
#include<cmath>
#include "tensor.hpp" 
#include <cuda_runtime.h>
#include <time.h>
#include <utility>
#include <type_traits>
#include <cmath>

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

double get_time();


void solve_cpu(Tensor2D<double>& A,
               Tensor2D<double>& A_new,
               Tensor2D<double>& f,
               double dx, double dy);
               
               
__global__ void solve_gpu(const double* A, double* A_new, const double* f, int nx, int ny, double dx, double dy);

__global__ void bc_gpu(double* A, int nx, int ny);

__global__ void error_gpu(double* A, double* A_exact, double* error, int nx, int ny);

__global__ void error_finalize_gpu(double* err_sum, int nx, int ny);

__global__ void swap_arrays(double* A, double* B, int n);

void rhs_gpu(Tensor2D<double>& f, double dx, double dy);

void exact_solution_gpu(Tensor2D<double>& A, double dx, double dy);
