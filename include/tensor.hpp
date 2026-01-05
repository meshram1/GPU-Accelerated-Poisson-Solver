#pragma once

#include <cstddef>
#include <cassert>
#include <cuda_runtime.h>
#include <utility>
using namespace std;

namespace cfd
{
#include<utility>
template <typename T>
class Tensor2D
{
private:
    std::size_t nx_, ny_;
    T* h_data_;
    T* d_data_;

private:
    void allocate_host()
    {
        h_data_ = new T[size()];
    }

    void free_host()
    {
        delete[] h_data_;
        h_data_ = nullptr;
    }

    void free_device()
    {
        if (d_data_)
        {
            cudaFree(d_data_);
            d_data_ = nullptr;
        }
    }

    void cleanup()
    {
        free_host();
        free_device();
    }

    void move_from(Tensor2D& other)
    {
        nx_ = other.nx_;
        ny_ = other.ny_;
        h_data_ = other.h_data_;
        d_data_ = other.d_data_;

        other.h_data_ = nullptr;
        other.d_data_ = nullptr;
    }

public:
    // Constructor
    Tensor2D(std::size_t nx, std::size_t ny)
        : nx_(nx), ny_(ny),
          h_data_(nullptr),
          d_data_(nullptr)
    {
        allocate_host();
    }
    
    // Destructor
    ~Tensor2D()
    {
        free_host();
        free_device();
    }

    // Disable copy
    Tensor2D(const Tensor2D&) = delete; // not allowing copy
    Tensor2D& operator=(const Tensor2D&) = delete; // 

    // Enable move
    Tensor2D(Tensor2D&& other) noexcept
    {
        move_from(other);
    }

    Tensor2D& operator=(Tensor2D&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            move_from(other);
        }
        return *this;
    }

    // Dimensions
    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }
    std::size_t size() const { return nx_ * ny_; }

    // Host access
    T* host_data() { return h_data_; }
    const T* host_data() const { return h_data_; }

    // Device access
    T* device_data() { return d_data_; }
    const T* device_data() const { return d_data_; }

    // Indexing (host only)
    T& operator()(std::size_t i, std::size_t j)
    {
        assert(h_data_);
        return h_data_[j * nx_ + i];
    }

    const T& operator()(std::size_t i, std::size_t j) const
    {
        assert(h_data_);
        return h_data_[j * nx_ + i];
    }

    // Allocate device memory
    void allocate_device()
    {
        if (!d_data_)
        {
            cudaMalloc(&d_data_, size() * sizeof(T));
        }
    }

    // Copy host → device
    void to_device()
    {
        assert(h_data_);
        allocate_device();
        cudaMemcpy(d_data_, h_data_,
                   size() * sizeof(T),
                   cudaMemcpyHostToDevice);
    }

    // Copy device → host
    void to_host()
    {
        assert(d_data_);
        cudaMemcpy(h_data_, d_data_,
                   size() * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }
    

    // Fill host data
    void fill(T value)
    {
        for (std::size_t i = 0; i < size(); ++i)
            h_data_[i] = value;
    }

};

}
