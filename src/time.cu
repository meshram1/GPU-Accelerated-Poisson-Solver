#include <iostream>
#include <fstream>
#include "../include/poisson.cuh"
#include "../include/tensor.hpp"
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

