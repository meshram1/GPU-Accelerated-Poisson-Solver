# GPU-Accelerated-Poisson-Solver
GPU-Accelerated Poisson Solver (CFD)
Overview

This project implements a GPU-accelerated solver for the 2D Poisson equation, a core subproblem in many Computational Fluid Dynamics (CFD) applications (e.g., pressure projection for incompressible flow).

The solver is written in C++ with CUDA and supports both CPU and GPU execution paths, enabling direct performance comparison. A custom tensor abstraction is used to manage host and device memory explicitly.

Features

Jacobi and Conjugate Gradient (CG) solvers for the Poisson equation

Matrix-free implementation using a 5-point finite-difference stencil

CUDA kernels for stencil application and residual computation

cuBLAS integration for vector operations (norms, dot products, axpy)

Ping-pong buffering to avoid unnecessary memory copies

CPU vs GPU benchmarking with measured speedups

Modular design suitable for extension to more advanced solvers

Numerical Method

The Poisson equation

∇
2
𝜙
=
𝑓
∇
2
ϕ=f

is discretized on a uniform 2D grid using second-order finite differences.

Two iterative solvers are provided:

Jacobi iteration (simple, fully parallel, GPU-friendly)

Conjugate Gradient (CG) (faster convergence, matrix-free, uses cuBLAS)

The Laplacian operator is applied without explicitly forming a matrix, which reduces memory usage and improves performance.

GPU Implementation

Each CUDA thread updates one interior grid point

Boundary conditions are handled separately

Device memory is reused across iterations

cuBLAS is used to keep reduction and vector operations on the GPU

This design minimizes host–device transfers and allows the solver to scale efficiently with grid size.

Performance

On tested grid sizes, the GPU implementation achieved:

~6× speedup per iteration compared to the CPU version

Up to ~13× end-to-end speedup for full solver runtime

Exact performance depends on grid resolution, convergence tolerance, and hardware.
