# bench: ours vs cupoisson

Wall-time benchmark between this solver and [cupoisson](https://github.com/3cHeLoN/cupoisson).
Both solve 2D Poisson on a regular grid. Ours uses Jacobi iteration on
the GPU with cuBLAS D-routines for the error reduction; cupoisson uses
a DST-based spectral solve. Both run as subprocesses, averaged over 5
runs.

## precision

Our solver is already double precision throughout (`Tensor2D<double>`,
`cublasDcopy`, `cublasDaxpy`, `cublasDnrm2`). No change needed on our
side.

cupoisson defaults to single precision. To switch:

1. In `third_party/cupoisson/csrc/precision.h`, uncomment
   `#define DOUBLE_PRECISION`.
2. In `third_party/cupoisson/csrc/Makefile`, uncomment the
   `NVCCPARMS += -arch sm_13` line and change `sm_13` to your card's
   architecture. `sm_13` is from 2008 and won't build on current CUDA.
   Use `sm_75` (T4 / RTX 20xx), `sm_80` (A100), `sm_86` (RTX 30xx), or
   `sm_89` (RTX 40xx).

then rebuild:

    cd third_party/cupoisson/csrc
    make clean
    make
    cd ../../..

## build our solver

From repo root. Substitute your own `sm_XX`.

    nvcc -arch=sm_75 -rdc=true -x cu -c src/main.cpp   -o src/main.o
    nvcc -arch=sm_75 -rdc=true -x cu -c src/poisson.cu -o src/poisson.o
    nvcc -arch=sm_75 -rdc=true -x cu -c src/time.cu    -o src/time.o
    nvcc -arch=sm_75 -rdc=true -lcublas -o run_gpu \
        src/main.o src/poisson.o src/time.o

## build the bench driver

    cd bench/csrc
    make
    cd ../..

## run

From repo root:

    ./bench/csrc/bench ./run_gpu ./third_party/cupoisson/csrc/cupoisson 5

Last argument is the number of timed runs (default 5). One warm-up
run of each is done first and discarded.

## things to know before reading the numbers

main.cpp currently runs the CPU Jacobi loop before the GPU loop. On a
300x300 grid that can dominate the wall time. For benchmarking, either
comment out the CPU `while` block in main.cpp, or have the bench parse
the "Total GPU time" line that main.cpp already prints.

cupoisson's DST is happiest when N = 2^k + 1 (257, 513, 1025). Our NX
and NY are 300 in main.cpp. For a fair head-to-head, either change
NX/NY to 257 or 513, or rebuild cupoisson with N near 300.

Subprocess wall time includes ~100-300 ms of CUDA context init per
launch. For solver-only numbers, parse the printed times from each
binary's stdout instead.
