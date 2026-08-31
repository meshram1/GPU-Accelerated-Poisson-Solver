# GPU-Accelerated Poisson Solver

Matrix-free CUDA solver for the 2D Poisson equation on a uniform grid, using a
5-point finite-difference stencil. Jacobi and conjugate gradient, with fused
kernels and mixed-precision variants, benchmarked against cuSPARSE and an FFT
direct solver.

**2.11× faster than cuSPARSE CG** at matched FP64 precision on a 2048² grid,
with identical iteration counts.

## Results

2048² grid, random right-hand side, tol 1e-4 relative residual, FP64 unless
noted. Every row runs the same CG algorithm; the FP64 rows converge in the
same 3362 iterations, so no row wins by solving an easier problem.

| Config | iters | ms | ms/iter | vs base |
|---|---:|---:|---:|---:|
| cuSPARSE CSR + cuBLAS, fp64 | 3362 | 15296.7 | 4.550 | 1.00× |
| matrix-free + cuBLAS, fp64 | 3362 | 10264.9 | 3.053 | 1.49× |
| matrix-free + fused, fp64 | 3370 | 7245.2 | 2.150 | **2.11×** |
| matrix-free + cuBLAS, mixed | 6626 | 9154.0 | 1.382 | 1.67× |
| matrix-free + fused, mixed | 6700 | 7223.4 | 1.078 | 2.12× |

cuSPARSE additionally pays 394.7 ms once to build and upload a 268 MB CSR
matrix — 20,938,768 nonzeros storing five distinct values. Matrix-free stores
nothing.

**Where the speedup comes from**

- **Matrix-free stencil — 1.49×.** CSR streams coefficients and column indices
  from memory every iteration; the stencil holds them as immediates and
  computes neighbour offsets arithmetically. Capped near 1.75× by Amdahl,
  since SpMV is only ~43% of per-iteration traffic.
- **Kernel fusion — 1.42× on top.** Merges the stencil with its dot product
  and the two AXPYs with theirs, cutting traffic ~30%, and keeps `alpha` and
  `beta` in device memory to remove two blocking host syncs per iteration.
- **Mixed precision — 2.0× per iteration, 1.00× overall.** FP32 inner
  iterations run at 1.078 ms vs 2.150 ms, but iterative refinement restarts CG
  each outer round and the iteration count doubles. The effects cancel at this
  tolerance — though mixed converges to 3.8e-07 where FP64 stops at 9.5e-05,
  roughly 250× tighter for the same wall time.

**Algorithm beats kernel tuning.** Jacobi needed 166,600 iterations at 300² to
reach the same tolerance; CG needs O(N) where Jacobi needs O(N²). At 2048²
that extrapolates to ~7.8 million Jacobi sweeps (~2.7 hours) against CG's
measured 3362 (7.2 s). Every kernel optimisation here together is worth 2.1×.

**An FFT solver is 229× faster.** `cuPoisson` solves the same grid directly in
31.6 ms (FP64), exactly rather than to 1e-4. For a rectangle with uniform
spacing and constant coefficients, that is the right algorithm and no kernel
tuning closes the gap. CG earns its place on variable coefficients, irregular
geometry, and warm-starting inside a timestep loop — none of which an FFT
solver can do.

## Structure

    .
    ├── build.sh                  compiles everything (use this, not raw nvcc)
    ├── sweep.sh                  Jacobi throughput across grid sizes
    ├── include/
    │   ├── tensor.hpp            host/device array with explicit transfers
    │   ├── poisson.cuh           stencil, residual, problem setup, declarations
    │   └── cuda_utlis.cpp
    ├── src/
    │   ├── main.cpp              drivers and benchmark harness
    │   ├── poisson.cu            Jacobi and residual kernels
    │   ├── cg.cu                 CG: plain, fused, and mixed-precision
    │   ├── cg_cusparse.cu        same CG loop over a CSR matrix (baseline)
    │   ├── time.cu
    │   ├── cublas_layer.cpp      unused
    │   └── cudann_conv.cpp       unused
    ├── bench/
    │   └── csrc/
    │       ├── Makefile
    │       └── bench.c
    ├── third_party/
    │   └── cupoisson/            (submodule) FFT direct solver, baseline
    └── README.md

## Clone

    git clone --recurse-submodules https://github.com/meshram1/GPU-Accelerated-Poisson-Solver.git
    cd GPU-Accelerated-Poisson-Solver

If you cloned without `--recurse-submodules`:

    git submodule update --init

Find your GPU arch and use it as `sm_XX` below:

    nvidia-smi --query-gpu=compute_cap --format=csv,noheader

## Build

    ./build.sh

Override the compiler or architecture if needed:

    ARCH=sm_86 NVCC=/usr/local/cuda/bin/nvcc ./build.sh

**Use `build.sh` rather than calling `nvcc` directly.** `nvcc` compiles *host*
code at `-O0` by default while giving device code `-O3`, which silently leaves
the CPU baseline as a debug build. That alone inflated the measured speedup by
roughly 13× before it was caught.

## Run

    ./run_gpu [N] [fixed_iters] [solver]

| argument | default | meaning |
|---|---|---|
| `N` | 300 | grid is N × N |
| `fixed_iters` | 0 | if > 0, run exactly this many iterations and skip the residual test |
| `solver` | `jacobi` | `jacobi`, `cg`, or `both` |

Jacobi, CPU vs GPU, converged at 300²:

    ./run_gpu

The full CG ladder — matrix-free, fused, mixed precision, cuSPARSE:

    RAND=1 ./run_gpu 2048 0 cg

Jacobi throughput across grid sizes (fixed iteration count, so per-sweep cost
is measured separately from convergence rate):

    ITERS=2000 SIZES="264 512 1024 2048 4096" ./sweep.sh

FFT baseline:

    ./third_party/cupoisson/csrc/cupoisson 2049

## Test problems

Default is a manufactured solution: `f = -2π²·sin(πx)sin(πy)`, whose exact
solution `u = sin(πx)sin(πy)` is known analytically.

`RAND=1` replaces it with a random right-hand side.

**`RAND=1` is required for any CG benchmark.** `sin(πx)sin(πy)` is an exact
eigenvector of the discrete Laplacian, so the right-hand side spans a
one-dimensional Krylov space and CG converges in *one iteration* — correct
behaviour, and useless for measuring anything about the inner loop. A random
RHS has energy in every eigenmode, which is also what a pressure-projection
right-hand side looks like mid-simulation. The reported L2 error against the
analytic solution is meaningless under `RAND=1`; the residual is the number
that matters there.

## Validation

- **Manufactured solution.** Converged L2 error of 4.58e-06 at 300² matches
  the predicted second-order discretisation error `(πh)²/12 × 0.5` to three
  significant figures.
- **Eigenvector termination.** With the `sin·sin` RHS, CG must finish in
  exactly one iteration. All CG variants do — a sharp check that every kernel
  implements the same operator.
- **Cross-implementation agreement.** CPU and GPU Jacobi converge identically:
  same 166,600 iterations, same residual, same L2 error to every printed digit.
  All five CG configurations agree on the answer despite differing reduction
  orders and precisions.
- **Residual-based stopping.** Convergence tests `‖b − Au‖ / ‖b‖`, never the
  error against a known solution, which would be unavailable in a real problem.
  The analytic solution is used only to report accuracy afterwards.

## Limitations

- **No multigrid comparison.** AmgX was not benchmarked. Algebraic multigrid
  converges in O(1) iterations against CG's O(N) and would be expected to win
  at scale. Closing that gap means a multigrid preconditioner — the natural
  next direction for this work.
- **cuSPARSE ran FP64 only.** The 2.11× headline is FP64-vs-FP64. cuSPARSE
  also has an FP32 path, so the mixed-precision rows should not be quoted
  against it without a precision-matched control.
- **Single GPU, single-threaded CPU baseline, 2D only.** No domain
  decomposition, no OpenMP baseline, no 3D. Shared-memory tiling was not
  implemented.
- **Dirichlet boundaries only.** Pressure projection in a closed domain needs
  homogeneous Neumann, which makes the operator singular and requires
  projecting the right-hand side to zero mean.
- **cuPoisson comparison caveats.** It was run with a uniform right-hand side
  against CG's random one — this does not change FFT cost, which is fixed, but
  a random RHS is harder for CG, so the gap somewhat flatters FFT. Interior
  sizes differ by one point (2047 vs 2046), which the sine transform requires.

## Build cupoisson (FP64 baseline)

The vendored copy ships in single precision and needs three edits before it is
comparable to the FP64 solver:

1. In `third_party/cupoisson/csrc/precision.h`, uncomment
   `#define DOUBLE_PRECISION`.

2. In `third_party/cupoisson/csrc/poisson.cu`, comment out the
   `cufftSetCompatibilityMode(...)` call (removed in CUDA 10+).

3. `main.c` takes the grid size from `argv[1]`. The sine transform needs an
   interior of 2^k − 1, so pass 513, 1025, 2049, …

Then, substituting your architecture:

    cd third_party/cupoisson/csrc
    rm -f *.o cupoisson
    g++ -Wall -O3 -I /usr/local/cuda/include -o main.o -c main.c
    nvcc -arch=sm_89 -O3 -o poisson.o -c poisson.cu
    nvcc -arch=sm_89 -O3 -o utils.o   -c utils.cu
    nvcc -arch=sm_89 -o cupoisson main.o poisson.o utils.o -lcufft -lcublas
    cd ../../..

The submodule's own `Makefile` assumes `nvcc` is on `PATH`; the explicit
commands above avoid that.

## Hardware

Results measured on an RTX 4050 Laptop (6 GB, sm_89) with a Ryzen 7 7735HS,
CUDA 12.1, under WSL2. FP64 runs at 1/64 the FP32 rate on consumer Ada, which
is why the mixed-precision path is worth exploring at all.
