#include<iostream>
#include<cmath>
#include "tensor.hpp" 
#include <cuda_runtime.h>

using namespace std;
using namespace cfd;

extern int lx;
extern int ly;
extern int nx;
extern int ny;



void rhs(Tensor2D<double>& f, double dx, double dy){
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

void boundary_conditions(Tensor2D<double>& A){
	for (std::size_t j = 0; j < A.ny(); ++j){
		A(0, j) = 0.0;
		A(A.nx()-1, j) = 0.0;
	}
	for (std::size_t i = 0; i < A.nx(); ++i){
		A(i, 0) = 0.0;
		A(i, A.ny()-1) = 0.0;
	} 
}

void exact_solution(Tensor2D<double>& A, double dx, double dy){
   const double pi = M_PI;
   for (std::size_t j = 0; j < A.ny(); ++j){
   	double y = j*dy;
		for (std::size_t i = 0; i < A.nx(); ++i){
			double x = i*dx;
			A(i, j) = std::sin(pi*x) * std::sin(pi*y);
		}
	}
}

void initial_guess(Tensor2D<double>& A, double value){
	A.fill(value);
}



double compute_error(Tensor2D<double>& A, Tensor2D<double>& A_exact){
	double error = 0.0;
	for (std::size_t j = 0; j < A.ny(); ++j){
		for (std::size_t i = 0; i < A.nx(); ++i){
			error += std::pow(A(i, j) - A_exact(i, j), 2);
		}
	}
	return std::sqrt(error / (A.nx() * A.ny()));
}
