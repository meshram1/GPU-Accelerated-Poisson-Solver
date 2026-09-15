#!/usr/bin/env bash
# Build the GPU Poisson solver.
#
# Host code is compiled at -O3 -march=native on purpose: nvcc defaults the HOST
# compiler to -O0, which would leave the CPU baseline as a debug build and make
# any CPU-vs-GPU speedup number meaningless.
set -e

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH=${ARCH:-sm_89}
OPT="-Xcompiler -O3 -Xcompiler -march=native"

cd "$(dirname "$0")"

"$NVCC" -arch=$ARCH -rdc=true $OPT -x cu -c src/main.cpp        -o src/main.o
"$NVCC" -arch=$ARCH -rdc=true $OPT -x cu -c src/poisson.cu      -o src/poisson.o
"$NVCC" -arch=$ARCH -rdc=true $OPT -x cu -c src/cg.cu           -o src/cg.o
"$NVCC" -arch=$ARCH -rdc=true $OPT -x cu -c src/cg_cusparse.cu  -o src/cg_cusparse.o
"$NVCC" -arch=$ARCH -rdc=true $OPT -x cu -c src/time.cu         -o src/time.o
"$NVCC" -arch=$ARCH -rdc=true $OPT -lcublas -lcusparse -o run_gpu \
        src/main.o src/poisson.o src/cg.o src/cg_cusparse.o src/time.o      
        

echo "BUILD_OK -> ./run_gpu"
