# GPU-Accelerated Poisson Solver

## structure

    .
    ├── include/
    │   ├── tensor.hpp
    │   ├── poisson.cuh
    │   └── cuda_utlis.cpp
    ├── src/
    │   ├── main.cpp
    │   ├── poisson.cu
    │   ├── time.cu
    │   ├── cublas_layer.cpp
    │   └── cudann_conv.cpp
    ├── bench/
    │   └── csrc/
    │       ├── Makefile
    │       └── bench.c
    ├── third_party/
    │   └── cupoisson/        (submodule)
    └── README.md

## clone

    git clone --recurse-submodules https://github.com/meshram1/GPU-Accelerated-Poisson-Solver.git
    cd GPU-Accelerated-Poisson-Solver

If you cloned without `--recurse-submodules`:

    git submodule update --init

Check your GPU arch:

    nvidia-smi --query-gpu=compute_cap --format=csv,noheader

Use that value as `sm_XX` below (e.g. `7.5` → `sm_75`, `8.9` → `sm_89`).

## build cupoisson (double precision)

1. In `third_party/cupoisson/csrc/precision.h`, uncomment
   `#define DOUBLE_PRECISION`.

2. In `third_party/cupoisson/csrc/Makefile`, uncomment the
   `NVCCPARMS += -arch sm_13` line and replace `sm_13` with your arch.

3. In `third_party/cupoisson/csrc/poisson.cu`, comment out the
   `cufftSetCompatibilityMode(...)` call (removed in CUDA 10+).

Then:

    cd third_party/cupoisson/csrc
    make clean
    make
    cd ../../..

## build our solver

    nvcc -arch=sm_89 -rdc=true -x cu -c src/main.cpp   -o src/main.o
    nvcc -arch=sm_89 -rdc=true -x cu -c src/poisson.cu -o src/poisson.o
    nvcc -arch=sm_89 -rdc=true -x cu -c src/time.cu    -o src/time.o
    nvcc -arch=sm_89 -rdc=true -lcublas -o run_gpu src/main.o src/poisson.o src/time.o

## run

ours:

    ./run_gpu

cupoisson:

    time ./third_party/cupoisson/csrc/cupoisson

## bench (optional)

    cd bench/csrc && make && cd ../..
    ./bench/csrc/bench ./run_gpu ./third_party/cupoisson/csrc/cupoisson 5
