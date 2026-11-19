/*
Header file for the CUDA helpers shared by every network on GPU.
Everything on the GPU is stored as float, since most GPUs are much slower with double.
*/

#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <algorithm>

#include <cuda_runtime.h>

#include "common.h"

// stops with the file and line if a CUDA call fails
#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err = (call);                                                   \
        if (err != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                         cudaGetErrorString(err));                                  \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

// kernel launches don't return an error, so check right after each one
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace gpu {

// threads per block, must be a power of 2 for block_reduce_sum
constexpr int BLOCK_SIZE = 256;

// width of the square tiles in gemm
constexpr int TILE_SIZE = 16;

// images fed forward at once when measuring accuracy or cost
constexpr int EVAL_BATCH_SIZE = 1000;

inline int num_blocks(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// array in GPU memory, copying it copies the array
template <typename T>
class DeviceBuffer {
    public:

    DeviceBuffer() {}

    explicit DeviceBuffer(size_t count) {
        this->resize(count);
    }

    explicit DeviceBuffer(const std::vector<T>& host) {
        this->upload(host);
    }

    DeviceBuffer(const DeviceBuffer& other) {
        this->resize(other.n);
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(ptr, other.ptr, n * sizeof(T), cudaMemcpyDeviceToDevice));
        }
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), n(other.n) {
        other.ptr = nullptr;
        other.n = 0;
    }

    // other is already a copy or a move, so swapping handles both
    DeviceBuffer& operator=(DeviceBuffer other) noexcept {
        std::swap(ptr, other.ptr);
        std::swap(n, other.n);
        return *this;
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    // contents are lost when the size changes
    void resize(size_t count) {
        if (count == n) {
            return;
        }
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
        }
        n = count;
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
    }

    void zero() {
        if (n > 0) {
            CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
        }
    }

    void upload(const std::vector<T>& host) {
        this->resize(host.size());
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(ptr, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    // first count elements
    std::vector<T> download(size_t count) const {
        std::vector<T> host(count);
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return host;
    }

    std::vector<T> download() const {
        return this->download(n);
    }

    T* data() { return ptr; }

    const T* data() const { return ptr; }

    size_t size() const { return n; }

    private:

    T* ptr = nullptr;
    size_t n = 0;
};

// images and labels uploaded to the GPU once
struct DeviceDataset {
    DeviceBuffer<float> images; // [n][dim]
    DeviceBuffer<int> labels; // [n]
    int n = 0;
    int dim = 0;

    const float* image(int i) const { return images.data() + static_cast<size_t>(i) * dim; }

    const int* label(int i) const { return labels.data() + i; }
};

DeviceDataset upload_dataset(const std::vector<std::pair<Vector, int>>& data);

// one-hot labels are stored as the digit
DeviceDataset upload_dataset(const std::vector<std::pair<Vector, Vector>>& data);

// copies images perm[start, start + count) into x and y, a shuffled mini-batch
void gather_batch(const DeviceDataset& data, const int* perm, int start, int count, float* x, int* y);

// C = alpha * op(A) * op(B) + beta * C, op(A) is M x K and op(B) is K x N, all row-major
void gemm(bool trans_a, bool trans_b, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);

// z[r][c] += b[c]
void add_bias(float* z, const float* b, int rows, int cols);

// out[c] = sum of column c
void column_sums(const float* m, int rows, int cols, float* out);

// x[i] *= y[i]
void multiply(float* x, const float* y, int n);

// w = decay * w - factor * nabla_w
void sgd_step(float* w, const float* nabla_w, int n, float decay, float factor);

double sum_squares(const float* x, int n);

// adds the number of rows whose largest output is the label to count
void count_correct(const float* output, const int* labels, int rows, int cols, int* count);

std::vector<float> flatten_matrix(const Matrix& m);

Matrix unflatten_matrix(const std::vector<float>& flat, int rows, int cols);

// device functions have to be in the header, each .cu file is compiled on its own

// sum of val over the block, the block needs exactly BLOCK_SIZE threads
template <typename T>
__device__ inline T block_reduce_sum(T val) {
    __shared__ T shared[BLOCK_SIZE];
    shared[threadIdx.x] = val;
    __syncthreads();

    // halve the active threads each step
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    T result = shared[0];
    __syncthreads(); // so shared can be reused
    return result;
}

// random number in [0, 1) from a seed and an index (splitmix64 hash)
__device__ inline float uniform_hash(unsigned long long seed, unsigned long long idx) {
    unsigned long long z = seed + (idx + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (z >> 40) * (1.0f / 16777216.0f);
}

__device__ inline float sigmoid_dev(float z) {
    return 1.0f / (1.0f + expf(-z));
}

}

#endif
