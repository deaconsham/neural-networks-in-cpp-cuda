/*
CUDA helpers shared by every network on GPU.
*/

#include "common.cuh"

namespace gpu {

// datasets

static int label_of(int y) {
    return y;
}

static int label_of(const Vector& y) {
    return argmax(y);
}

template <typename Label>
static DeviceDataset upload_impl(const std::vector<std::pair<Vector, Label>>& data) {
    DeviceDataset d;
    d.n = data.size();
    d.dim = d.n > 0 ? data[0].first.size() : 0;

    std::vector<float> images;
    std::vector<int> labels;
    images.reserve(static_cast<size_t>(d.n) * d.dim);
    for (int i = 0; i < d.n; i++) {
        images.insert(images.end(), data[i].first.begin(), data[i].first.end());
        labels.push_back(label_of(data[i].second));
    }

    d.images.upload(images);
    d.labels.upload(labels);
    return d;
}

DeviceDataset upload_dataset(const std::vector<std::pair<Vector, int>>& data) {
    return upload_impl(data);
}

DeviceDataset upload_dataset(const std::vector<std::pair<Vector, Vector>>& data) {
    return upload_impl(data);
}

// one thread per pixel of the mini-batch
__global__ void gather_kernel(const float* images, const int* labels, const int* perm, int start, int count, int dim, float* x, int* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * dim) {
        return;
    }
    int row = i / dim;
    int col = i % dim;
    int src = perm[start + row];

    x[i] = images[static_cast<size_t>(src) * dim + col];
    if (col == 0) {
        y[row] = labels[src];
    }
}

void gather_batch(const DeviceDataset& data, const int* perm, int start, int count, float* x, int* y) {
    gather_kernel<<<num_blocks(count * data.dim), BLOCK_SIZE>>>(data.images.data(), data.labels.data(), perm, start, count, data.dim, x, y);
    CUDA_CHECK_LAUNCH();
}

// matrix multiplication

// each block loads a tile of op(A) and op(B) into shared memory, so each value read is used TILE_SIZE times
template <bool TRANS_A, bool TRANS_B>
__global__ void gemm_kernel(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE_SIZE) {
        // op(A)[row][a_k], zero past the edge
        int a_k = t + threadIdx.x;
        float a_val = 0.0f;
        if (row < M && a_k < K) {
            a_val = TRANS_A ? A[a_k * M + row] : A[row * K + a_k];
        }
        a_tile[threadIdx.y][threadIdx.x] = a_val;

        // op(B)[b_k][col], zero past the edge
        int b_k = t + threadIdx.y;
        float b_val = 0.0f;
        if (b_k < K && col < N) {
            b_val = TRANS_B ? B[col * K + b_k] : B[b_k * N + col];
        }
        b_tile[threadIdx.y][threadIdx.x] = b_val;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // C isn't read when beta is 0, it may not be initialized
        int idx = row * N + col;
        C[idx] = beta == 0.0f ? alpha * sum : alpha * sum + beta * C[idx];
    }
}

void gemm(bool trans_a, bool trans_b, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    if (!trans_a && !trans_b) {
        gemm_kernel<false, false><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    } else if (!trans_a && trans_b) {
        gemm_kernel<false, true><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    } else if (trans_a && !trans_b) {
        gemm_kernel<true, false><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    } else {
        gemm_kernel<true, true><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }
    CUDA_CHECK_LAUNCH();
}

// element-wise

__global__ void add_bias_kernel(float* z, const float* b, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        z[i] += b[i % cols];
    }
}

void add_bias(float* z, const float* b, int rows, int cols) {
    add_bias_kernel<<<num_blocks(rows * cols), BLOCK_SIZE>>>(z, b, rows, cols);
    CUDA_CHECK_LAUNCH();
}

// one thread per column, rows is the batch size so the loop is short
__global__ void column_sums_kernel(const float* m, int rows, int cols, float* out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) {
        return;
    }
    float sum = 0.0f;
    for (int r = 0; r < rows; r++) {
        sum += m[r * cols + c];
    }
    out[c] = sum;
}

void column_sums(const float* m, int rows, int cols, float* out) {
    column_sums_kernel<<<num_blocks(cols), BLOCK_SIZE>>>(m, rows, cols, out);
    CUDA_CHECK_LAUNCH();
}

__global__ void multiply_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= y[i];
    }
}

void multiply(float* x, const float* y, int n) {
    multiply_kernel<<<num_blocks(n), BLOCK_SIZE>>>(x, y, n);
    CUDA_CHECK_LAUNCH();
}

__global__ void sgd_step_kernel(float* w, const float* nabla_w, int n, float decay, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = decay * w[i] - factor * nabla_w[i];
    }
}

void sgd_step(float* w, const float* nabla_w, int n, float decay, float factor) {
    sgd_step_kernel<<<num_blocks(n), BLOCK_SIZE>>>(w, nabla_w, n, decay, factor);
    CUDA_CHECK_LAUNCH();
}

// reductions

// each block sums part of x, the partial sums are added on the CPU in double
__global__ void sum_squares_kernel(const float* x, int n, float* partial) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        sum += x[i] * x[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sum;
    }
}

double sum_squares(const float* x, int n) {
    int blocks = std::min(num_blocks(n), 1024);
    if (blocks == 0) {
        return 0.0;
    }
    DeviceBuffer<float> partial(blocks);
    sum_squares_kernel<<<blocks, BLOCK_SIZE>>>(x, n, partial.data());
    CUDA_CHECK_LAUNCH();

    std::vector<float> host = partial.download();
    double total = 0.0;
    for (int i = 0; i < host.size(); i++) {
        total += host[i];
    }
    return total;
}

// one thread per image, if the network's guess matches the label, increment count
__global__ void count_correct_kernel(const float* output, const int* labels, int rows, int cols, int* count) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float* out = output + r * cols;
    int max_idx = 0;
    for (int c = 1; c < cols; c++) {
        if (out[c] > out[max_idx]) {
            max_idx = c;
        }
    }
    if (max_idx == labels[r]) {
        atomicAdd(count, 1);
    }
}

void count_correct(const float* output, const int* labels, int rows, int cols, int* count) {
    count_correct_kernel<<<num_blocks(rows), BLOCK_SIZE>>>(output, labels, rows, cols, count);
    CUDA_CHECK_LAUNCH();
}

// conversions

std::vector<float> flatten_matrix(const Matrix& m) {
    std::vector<float> arr;
    for (int i = 0; i < m.size(); i++) {
        arr.insert(arr.end(), m[i].begin(), m[i].end());
    }
    return arr;
}

Matrix unflatten_matrix(const std::vector<float>& flat, int rows, int cols) {
    Matrix m(rows, Vector(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m[i][j] = flat[i * cols + j];
        }
    }
    return m;
}

}
