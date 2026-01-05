# GPU-Accelerated-Poisson-Solver

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
Performance

On tested grid sizes, the GPU implementation achieved:
~6× speedup per iteration compared to the CPU version
Up to ~13× end-to-end speedup for full solver runtime
Exact performance depends on grid resolution, convergence tolerance, and hardware.
