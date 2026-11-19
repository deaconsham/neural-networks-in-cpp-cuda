/*
Final convolutional neural network from Chapter 6 (network3.py) on GPU (CUDA).
Architecture: convolutional-pooling layers, fully-connected layers and a softmax output layer.
Uses ReLU activations, dropout, L2 regularization and the log-likelihood cost.
Gradients are calculated using backpropagation.
*/

#include "conv_network.cuh"

namespace gpu {

// convolutional + max-pooling kernels

// convolution of image b with filter f at (r, c)
__device__ inline float conv_at(const float* x, const float* w, const ConvShape& s, int b, int f, int r, int c) {
    const float* image = x + b * s.in_channels * s.in_rows * s.in_cols;
    const float* filter = w + f * s.in_channels * s.filter_rows * s.filter_cols;

    float sum = 0.0f;
    for (int ch = 0; ch < s.in_channels; ch++) {
        for (int kr = 0; kr < s.filter_rows; kr++) {
            for (int kc = 0; kc < s.filter_cols; kc++) {
                sum += filter[(ch * s.filter_rows + kr) * s.filter_cols + kc] * image[(ch * s.in_rows + r + kr) * s.in_cols + c + kc];
            }
        }
    }
    return sum;
}

// one thread per pooled output, i is its index in z, max_idx and out
__global__ void conv_pool_forward_kernel(const float* x, const float* w, const float* biases, ConvShape s, int batch_size, float* z, int* max_idx, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * s.num_filters * s.out_rows * s.out_cols) {
        return;
    }
    int pj = i % s.out_cols;
    int pi = (i / s.out_cols) % s.out_rows;
    int f = (i / (s.out_cols * s.out_rows)) % s.num_filters;
    int b = i / (s.out_cols * s.out_rows * s.num_filters);

    // max pooling, then bias and activation

    float max_val = -INFINITY;
    int best = 0;
    for (int di = 0; di < s.pool_size; di++) {
        for (int dj = 0; dj < s.pool_size; dj++) {
            int r = pi * s.pool_size + di;
            int c = pj * s.pool_size + dj;
            float conv = conv_at(x, w, s, b, f, r, c);
            if (conv > max_val) {
                max_val = conv;
                best = r * s.conv_cols + c;
            }
        }
    }
    max_idx[i] = best;
    z[i] = max_val + biases[f];
    out[i] = fmaxf(z[i], 0.0f);
}

// dz = delta * relu'(z), also used by the fully-connected layers
__global__ void relu_backward_kernel(const float* delta, const float* z, int n, float* dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dz[i] = z[i] > 0.0f ? delta[i] : 0.0f;
    }
}

// one block per filter, sums dz over the batch
__global__ void conv_bias_grad_kernel(const float* dz, ConvShape s, int batch_size, float* nabla_b) {
    int f = blockIdx.x;
    int per_image = s.out_rows * s.out_cols;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < batch_size * per_image; k += blockDim.x) {
        int b = k / per_image;
        int p = k % per_image;
        sum += dz[(b * s.num_filters + f) * per_image + p];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        nabla_b[f] = sum;
    }
}

// one thread per weight
// only the max of each pooling window received input, so only it gets gradient
__global__ void conv_weight_grad_kernel(const float* input, const float* dz, const int* max_idx, ConvShape s, int batch_size, float* nabla_w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.num_filters * s.in_channels * s.filter_rows * s.filter_cols) {
        return;
    }
    int kc = i % s.filter_cols;
    int kr = (i / s.filter_cols) % s.filter_rows;
    int ch = (i / (s.filter_cols * s.filter_rows)) % s.in_channels;
    int f = i / (s.filter_cols * s.filter_rows * s.in_channels);

    int per_image = s.out_rows * s.out_cols;
    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        const float* image = input + (b * s.in_channels + ch) * s.in_rows * s.in_cols;
        for (int p = 0; p < per_image; p++) {
            int o = (b * s.num_filters + f) * per_image + p;
            if (dz[o] == 0.0f) {
                continue;
            }
            int r = max_idx[o] / s.conv_cols;
            int c = max_idx[o] % s.conv_cols;
            sum += dz[o] * image[(r + kr) * s.in_cols + c + kc];
        }
    }
    nabla_w[i] = sum;
}

// one thread per pooled output
// windows of neighbouring outputs overlap, so the adds have to be atomic
__global__ void conv_input_grad_kernel(const float* dz, const int* max_idx, const float* w, ConvShape s, int batch_size, float* delta_input) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * s.num_filters * s.out_rows * s.out_cols) {
        return;
    }
    if (dz[i] == 0.0f) {
        return;
    }
    int f = (i / (s.out_cols * s.out_rows)) % s.num_filters;
    int b = i / (s.out_cols * s.out_rows * s.num_filters);
    int r = max_idx[i] / s.conv_cols;
    int c = max_idx[i] % s.conv_cols;

    const float* filter = w + f * s.in_channels * s.filter_rows * s.filter_cols;
    float* image = delta_input + b * s.in_channels * s.in_rows * s.in_cols;

    for (int ch = 0; ch < s.in_channels; ch++) {
        for (int kr = 0; kr < s.filter_rows; kr++) {
            for (int kc = 0; kc < s.filter_cols; kc++) {
                atomicAdd(&image[(ch * s.in_rows + r + kr) * s.in_cols + c + kc], dz[i] * filter[(ch * s.filter_rows + kr) * s.filter_cols + kc]);
            }
        }
    }
}

// fully-connected and softmax kernels

__global__ void relu_kernel(const float* z, int n, float* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = fmaxf(z[i], 0.0f);
    }
}

// randomly drop inputs, each one is kept with probability keep
__global__ void dropout_kernel(const float* x, int n, float keep, unsigned long long seed, float* mask, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        mask[i] = uniform_hash(seed, i) < keep ? 1.0f : 0.0f;
        out[i] = x[i] * mask[i];
    }
}

// one thread per image, subtract the max for numerical stability
__global__ void softmax_kernel(const float* z, int rows, int cols, float* out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float* zr = z + r * cols;
    float* arr = out + r * cols;

    float max_val = zr[0];
    for (int c = 1; c < cols; c++) {
        max_val = fmaxf(max_val, zr[c]);
    }

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        arr[c] = expf(zr[c] - max_val);
        sum += arr[c];
    }
    for (int c = 0; c < cols; c++) {
        arr[c] /= sum;
    }
}

// log-likelihood cost with softmax gives delta = a - y
__global__ void softmax_delta_kernel(const float* a, const int* labels, int rows, int cols, float* dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        dz[i] = a[i] - (i % cols == labels[i / cols] ? 1.0f : 0.0f);
    }
}

// convolutional + max-pooling layer

ConvPoolLayer::ConvPoolLayer(const std::vector<int>& image_shape, const std::vector<int>& filter_shape, int pool_size, std::mt19937& gen) {
    assert(image_shape[0] == filter_shape[1] && "Image channels must match filter channels");

    num_filters = filter_shape[0];
    in_channels = filter_shape[1];
    filter_rows = filter_shape[2];
    filter_cols = filter_shape[3];

    in_rows = image_shape[1];
    in_cols = image_shape[2];

    this->pool_size = pool_size;

    // valid convolution shrinks the image, pooling divides it
    conv_rows = in_rows - filter_rows + 1;
    conv_cols = in_cols - filter_cols + 1;
    out_rows = conv_rows / pool_size;
    out_cols = conv_cols / pool_size;

    // weights ~ N(0, 1/n_out), biases ~ N(0, 1)
    double n_out = static_cast<double>(num_filters * filter_rows * filter_cols) / (pool_size * pool_size);
    std::normal_distribution<double> w_dis(0.0, std::sqrt(1.0 / n_out));
    std::normal_distribution<double> b_dis(0.0, 1.0);

    std::vector<float> w(num_filters * in_channels * filter_rows * filter_cols);
    for (int i = 0; i < w.size(); i++) {
        w[i] = w_dis(gen);
    }

    std::vector<float> b(num_filters);
    for (int f = 0; f < num_filters; f++) {
        b[f] = b_dis(gen);
    }

    weights.upload(w);
    biases.upload(b);
    nabla_w.resize(w.size());
    nabla_b.resize(b.size());
}

ConvShape ConvPoolLayer::shape() const {
    return {in_channels, in_rows, in_cols, num_filters, filter_rows, filter_cols, pool_size, conv_rows, conv_cols, out_rows, out_cols};
}

int ConvPoolLayer::input_size() const {
    return in_channels * in_rows * in_cols;
}

int ConvPoolLayer::output_size() const {
    return num_filters * out_rows * out_cols;
}

void ConvPoolLayer::ensure_capacity(int batch_size) {
    if (batch_size <= capacity) {
        return;
    }
    capacity = batch_size;
    z.resize(capacity * this->output_size());
    max_idx.resize(capacity * this->output_size());
    output.resize(capacity * this->output_size());
    dz.resize(capacity * this->output_size());
    delta_input.resize(capacity * this->input_size());
}

const float* ConvPoolLayer::feed_forward(const float* x, int batch_size) {
    this->ensure_capacity(batch_size);
    input = x;

    int n = batch_size * this->output_size();
    conv_pool_forward_kernel<<<num_blocks(n), BLOCK_SIZE>>>(x, weights.data(), biases.data(), this->shape(), batch_size, z.data(), max_idx.data(), output.data());
    CUDA_CHECK_LAUNCH();
    return output.data();
}

const float* ConvPoolLayer::backpropagation(const float* delta, int batch_size, bool compute_delta_input) {
    int n = batch_size * this->output_size();
    ConvShape s = this->shape();

    relu_backward_kernel<<<num_blocks(n), BLOCK_SIZE>>>(delta, z.data(), n, dz.data());
    CUDA_CHECK_LAUNCH();

    conv_bias_grad_kernel<<<num_filters, BLOCK_SIZE>>>(dz.data(), s, batch_size, nabla_b.data());
    CUDA_CHECK_LAUNCH();

    conv_weight_grad_kernel<<<num_blocks(weights.size()), BLOCK_SIZE>>>(input, dz.data(), max_idx.data(), s, batch_size, nabla_w.data());
    CUDA_CHECK_LAUNCH();

    if (!compute_delta_input) {
        return nullptr;
    }

    CUDA_CHECK(cudaMemset(delta_input.data(), 0, batch_size * this->input_size() * sizeof(float)));
    conv_input_grad_kernel<<<num_blocks(n), BLOCK_SIZE>>>(dz.data(), max_idx.data(), weights.data(), s, batch_size, delta_input.data());
    CUDA_CHECK_LAUNCH();
    return delta_input.data();
}

void ConvPoolLayer::update(double learning_rate, int mini_batch_size, double weight_decay) {
    double factor = learning_rate / mini_batch_size;
    double decay = 1.0 - learning_rate * weight_decay;

    sgd_step(weights.data(), nabla_w.data(), weights.size(), decay, factor);
    sgd_step(biases.data(), nabla_b.data(), biases.size(), 1.0f, factor);
}

// fully-connected layer

FullyConnectedLayer::FullyConnectedLayer(int n_in, int n_out, double p_dropout) {
    this->n_in = n_in;
    this->n_out = n_out;
    this->p_dropout = p_dropout;

    weights.resize(n_out * n_in);
    biases.resize(n_out);
    weights.zero();
    biases.zero();

    nabla_w.resize(n_out * n_in);
    nabla_b.resize(n_out);
}

FullyConnectedLayer::FullyConnectedLayer(int n_in, int n_out, std::mt19937& gen, double p_dropout)
    : FullyConnectedLayer(n_in, n_out, p_dropout)
{
    // weights ~ N(0, 1/n_out), biases ~ N(0, 1)
    std::normal_distribution<double> w_dis(0.0, std::sqrt(1.0 / n_out));
    std::normal_distribution<double> b_dis(0.0, 1.0);

    std::vector<float> w(n_out * n_in);
    std::vector<float> b(n_out);
    for (int j = 0; j < n_out; j++) {
        b[j] = b_dis(gen);
        for (int k = 0; k < n_in; k++) {
            w[j * n_in + k] = w_dis(gen);
        }
    }
    weights.upload(w);
    biases.upload(b);
}

void FullyConnectedLayer::ensure_capacity(int batch_size) {
    if (batch_size <= capacity) {
        return;
    }
    capacity = batch_size;
    input.resize(capacity * n_in);
    mask.resize(capacity * n_in);
    delta_input.resize(capacity * n_in);
    z.resize(capacity * n_out);
    output.resize(capacity * n_out);
    dz.resize(capacity * n_out);
}

void FullyConnectedLayer::weighted_input(const float* x, int batch_size, float scale) {
    this->ensure_capacity(batch_size);

    // z = scale * x * w^T + b
    gemm(false, true, batch_size, n_out, n_in, scale, x, weights.data(), 0.0f, z.data());
    add_bias(z.data(), biases.data(), batch_size, n_out);
}

void FullyConnectedLayer::weighted_input_dropout(const float* x, int batch_size, unsigned long long seed) {
    this->ensure_capacity(batch_size);

    // randomly drop inputs to this layer
    int n = batch_size * n_in;
    dropout_kernel<<<num_blocks(n), BLOCK_SIZE>>>(x, n, 1.0f - p_dropout, seed, mask.data(), input.data());
    CUDA_CHECK_LAUNCH();

    this->weighted_input(input.data(), batch_size, 1.0f);
}

const float* FullyConnectedLayer::backpropagate_dz(int batch_size) {
    // gradients summed over the batch
    column_sums(dz.data(), batch_size, n_out, nabla_b.data());
    gemm(true, false, n_out, n_in, batch_size, 1.0f, dz.data(), input.data(), 0.0f, nabla_w.data());

    // transpose weight multiplication
    gemm(false, false, batch_size, n_in, n_out, 1.0f, dz.data(), weights.data(), 0.0f, delta_input.data());

    // dropped inputs did not affect the output
    multiply(delta_input.data(), mask.data(), batch_size * n_in);
    return delta_input.data();
}

const float* FullyConnectedLayer::feed_forward(const float* x, int batch_size) {
    // no dropout at inference, so scale weights by the keep probability instead
    this->weighted_input(x, batch_size, 1.0f - p_dropout);

    relu_kernel<<<num_blocks(batch_size * n_out), BLOCK_SIZE>>>(z.data(), batch_size * n_out, output.data());
    CUDA_CHECK_LAUNCH();
    return output.data();
}

const float* FullyConnectedLayer::feed_forward_dropout(const float* x, int batch_size, unsigned long long seed) {
    this->weighted_input_dropout(x, batch_size, seed);

    relu_kernel<<<num_blocks(batch_size * n_out), BLOCK_SIZE>>>(z.data(), batch_size * n_out, output.data());
    CUDA_CHECK_LAUNCH();
    return output.data();
}

const float* FullyConnectedLayer::backpropagation(const float* delta, int batch_size) {
    relu_backward_kernel<<<num_blocks(batch_size * n_out), BLOCK_SIZE>>>(delta, z.data(), batch_size * n_out, dz.data());
    CUDA_CHECK_LAUNCH();
    return this->backpropagate_dz(batch_size);
}

void FullyConnectedLayer::update(double learning_rate, int mini_batch_size, double weight_decay) {
    double factor = learning_rate / mini_batch_size;
    double decay = 1.0 - learning_rate * weight_decay;

    sgd_step(weights.data(), nabla_w.data(), weights.size(), decay, factor);
    sgd_step(biases.data(), nabla_b.data(), biases.size(), 1.0f, factor);
}

// softmax output layer

SoftmaxLayer::SoftmaxLayer(int n_in, int n_out, double p_dropout)
    : FullyConnectedLayer(n_in, n_out, p_dropout)    // softmax weights and biases start at zero
{
}

const float* SoftmaxLayer::feed_forward(const float* x, int batch_size) {
    this->weighted_input(x, batch_size, 1.0f - p_dropout);

    softmax_kernel<<<num_blocks(batch_size), BLOCK_SIZE>>>(z.data(), batch_size, n_out, output.data());
    CUDA_CHECK_LAUNCH();
    return output.data();
}

const float* SoftmaxLayer::feed_forward_dropout(const float* x, int batch_size, unsigned long long seed) {
    this->weighted_input_dropout(x, batch_size, seed);

    softmax_kernel<<<num_blocks(batch_size), BLOCK_SIZE>>>(z.data(), batch_size, n_out, output.data());
    CUDA_CHECK_LAUNCH();
    return output.data();
}

const float* SoftmaxLayer::backpropagation(const int* labels, int batch_size) {
    softmax_delta_kernel<<<num_blocks(batch_size * n_out), BLOCK_SIZE>>>(output.data(), labels, batch_size, n_out, dz.data());
    CUDA_CHECK_LAUNCH();
    return this->backpropagate_dz(batch_size);
}

// network

ConvNetwork::ConvNetwork(const std::vector<ConvPoolLayer>& conv_layers, const std::vector<FullyConnectedLayer>& fc_layers, const SoftmaxLayer& softmax_layer)
    : conv_layers(conv_layers), fc_layers(fc_layers), softmax_layer(softmax_layer)
{
    std::random_device rd;
    gen.seed(rd());
}

unsigned long long ConvNetwork::random_seed() {
    return (static_cast<unsigned long long>(gen()) << 32) | gen();
}

const float* ConvNetwork::conv_feed_forward(const float* x, int batch_size) {
    // [batch][filter][row][col] is already flattened for the fully-connected layers
    const float* image = x;
    for (int i = 0; i < conv_layers.size(); i++) {
        image = conv_layers[i].feed_forward(image, batch_size);
    }
    return image;
}

const float* ConvNetwork::feed_forward(const float* x, int batch_size) {
    const float* arr = this->conv_feed_forward(x, batch_size);
    for (int i = 0; i < fc_layers.size(); i++) {
        arr = fc_layers[i].feed_forward(arr, batch_size);
    }
    return softmax_layer.feed_forward(arr, batch_size);
}

void ConvNetwork::sgd(
    std::vector<std::pair<Vector, int>>& training_data,
    int epochs,
    int mini_batch_size,
    double learning_rate,
    const std::vector<std::pair<Vector, int>>& validation_data,
    const std::vector<std::pair<Vector, int>>& test_data,
    double lmbda)
{
    int n = training_data.size();
    int num_training_batches = n / mini_batch_size;

    // data is copied to the GPU once, then an index list is shuffled instead of the images
    DeviceDataset train = upload_dataset(training_data);
    DeviceDataset validation = upload_dataset(validation_data);
    DeviceDataset test = upload_dataset(test_data);

    std::vector<int> perm(n);
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    DeviceBuffer<int> perm_dev(n);

    DeviceBuffer<float> batch_x(mini_batch_size * train.dim);
    DeviceBuffer<int> batch_y(mini_batch_size);

    double best_validation_accuracy = 0.0;
    int best_iteration = 0;
    double test_accuracy = 0.0;

    for (int i = 0; i < epochs; i++) {
        std::shuffle(perm.begin(), perm.end(), gen);
        perm_dev.upload(perm);

        for (int j = 0; j < num_training_batches; j++) {
            int iteration = num_training_batches * i + j;
            if (iteration % 1000 == 0) {
                std::cout << "Training mini-batch number " << iteration << std::endl;
            }

            gather_batch(train, perm_dev.data(), j * mini_batch_size, mini_batch_size, batch_x.data(), batch_y.data());
            this->update_mini_batch(batch_x.data(), batch_y.data(), mini_batch_size, learning_rate, lmbda, num_training_batches);
        }

        double validation_accuracy = this->accuracy(validation) * 100.0;
        std::cout << "Epoch " << i + 1 << ": validation accuracy " << validation_accuracy << "%" << std::endl;

        // only check the test set when validation improves
        if (validation_accuracy >= best_validation_accuracy) {
            std::cout << "This is the best validation accuracy to date." << std::endl;
            best_validation_accuracy = validation_accuracy;
            best_iteration = num_training_batches * (i + 1) - 1;
            test_accuracy = this->accuracy(test) * 100.0;
            std::cout << "The corresponding test accuracy is " << test_accuracy << "%" << std::endl;
        }
    }

    std::cout << "Finished training network." << std::endl;
    std::cout << "Best validation accuracy of " << best_validation_accuracy << "% obtained at iteration " << best_iteration << std::endl;
    std::cout << "Corresponding test accuracy of " << test_accuracy << "%" << std::endl;
}

void ConvNetwork::update_mini_batch(const float* x, const int* y, int batch_size, double learning_rate, double lmbda, int num_training_batches) {
    // gradients are summed over the batch, so they don't need to be zeroed first
    this->backpropagation(x, y, batch_size);

    // L2 term 0.5 * lmbda * sum(w^2) / num_training_batches, as in network3.py
    double weight_decay = lmbda / num_training_batches;

    for (int i = 0; i < conv_layers.size(); i++) {
        conv_layers[i].update(learning_rate, batch_size, weight_decay);
    }
    for (int i = 0; i < fc_layers.size(); i++) {
        fc_layers[i].update(learning_rate, batch_size, weight_decay);
    }
    softmax_layer.update(learning_rate, batch_size, weight_decay);
}

void ConvNetwork::backpropagation(const float* x, const int* y, int batch_size) {

    // forward pass with dropout

    const float* arr = this->conv_feed_forward(x, batch_size);
    for (int i = 0; i < fc_layers.size(); i++) {
        arr = fc_layers[i].feed_forward_dropout(arr, batch_size, this->random_seed());
    }
    softmax_layer.feed_forward_dropout(arr, batch_size, this->random_seed());

    // backward pass

    const float* delta = softmax_layer.backpropagation(y, batch_size);

    for (int i = fc_layers.size() - 1; i >= 0; i--) {
        delta = fc_layers[i].backpropagation(delta, batch_size);
    }

    for (int i = conv_layers.size() - 1; i >= 0; i--) {
        delta = conv_layers[i].backpropagation(delta, batch_size, i > 0);
    }
}

double ConvNetwork::accuracy(const DeviceDataset& data) {
    DeviceBuffer<int> count(1);
    count.zero();

    for (int start = 0; start < data.n; start += EVAL_BATCH_SIZE) {
        int batch_size = std::min(EVAL_BATCH_SIZE, data.n - start);
        const float* out = this->feed_forward(data.image(start), batch_size);
        count_correct(out, data.label(start), batch_size, softmax_layer.n_out, count.data());
    }
    return static_cast<double>(count.download()[0]) / data.n;
}

double ConvNetwork::accuracy(const std::vector<std::pair<Vector, int>>& data) {
    return this->accuracy(upload_dataset(data));
}

}
