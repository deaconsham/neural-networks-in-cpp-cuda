/*
SGD learning algorithm for a MNIST-trained neural network on GPU (CUDA).
Each mini-batch is one matrix with a row per image, instead of one image at a time.
Gradients are calculated using backpropagation.
*/

#include "network.cuh"

namespace gpu {

// kernels

// z += b, then a = sigmoid(z)
__global__ void bias_sigmoid_kernel(float* z, const float* b, int rows, int cols, float* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        z[i] += b[i % cols];
        a[i] = sigmoid_dev(z[i]);
    }
}

// delta *= sigmoid'(z)
__global__ void sigmoid_prime_kernel(float* delta, const float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sigmoid_dev(z[i]);
        delta[i] *= s * (1.0f - s);
    }
}

// quadratic: (a - y) * sigmoid'(z), cross-entropy: a - y
__global__ void output_delta_kernel(bool cross_entropy, const float* a, const int* labels, int rows, int cols, float* delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) {
        return;
    }
    float y = (i % cols == labels[i / cols]) ? 1.0f : 0.0f;
    float d = a[i] - y;
    if (!cross_entropy) {
        d *= a[i] * (1.0f - a[i]);
    }
    delta[i] = d;
}

// ln(1 + e^x) without overflowing
__device__ inline float softplus(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

// one thread per image
// cross-entropy uses -ln(a) = softplus(-z) and -ln(1 - a) = softplus(z), since in float a = 1.0 once z > ~17
__global__ void sample_cost_kernel(bool cross_entropy, const float* z, const float* a, const int* labels, int rows, int cols, float* cost) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    float c = 0.0f;
    for (int j = 0; j < cols; j++) {
        int i = r * cols + j;
        bool is_label = (j == labels[r]);
        if (cross_entropy) {
            c += is_label ? softplus(-z[i]) : softplus(z[i]);
        } else {
            float error = a[i] - (is_label ? 1.0f : 0.0f);
            c += 0.5f * error * error;
        }
    }
    cost[r] = c;
}

// sigmoid layers

SigmoidMLP::SigmoidMLP(const std::vector<int>& sizes) {
    num_layers = sizes.size();
    this->sizes = sizes;

    for (int i = 0; i < num_layers - 1; i++) {
        biases.push_back(DeviceBuffer<float>(sizes[i + 1]));
        weights.push_back(DeviceBuffer<float>(sizes[i + 1] * sizes[i]));
        nabla_b.push_back(DeviceBuffer<float>(sizes[i + 1]));
        nabla_w.push_back(DeviceBuffer<float>(sizes[i + 1] * sizes[i]));
    }

    act_ptrs.resize(num_layers, nullptr);
    activations.resize(num_layers);
    zs.resize(num_layers - 1);
    deltas.resize(num_layers - 1);
}

Matrix SigmoidMLP::get_biases() const {
    Matrix b;
    for (int i = 0; i < biases.size(); i++) {
        std::vector<float> arr = biases[i].download();
        b.push_back(Vector(arr.begin(), arr.end()));
    }
    return b;
}

Tensor SigmoidMLP::get_weights() const {
    Tensor w;
    for (int i = 0; i < weights.size(); i++) {
        w.push_back(unflatten_matrix(weights[i].download(), sizes[i + 1], sizes[i]));
    }
    return w;
}

void SigmoidMLP::set_parameters(const Matrix& b, const Tensor& w) {
    for (int i = 0; i < num_layers - 1; i++) {
        biases[i].upload(std::vector<float>(b[i].begin(), b[i].end()));
        weights[i].upload(flatten_matrix(w[i]));
    }
}

void SigmoidMLP::ensure_capacity(int batch_size) {
    if (batch_size <= capacity) {
        return;
    }
    capacity = batch_size;
    for (int i = 1; i < num_layers; i++) {
        activations[i].resize(capacity * sizes[i]);
    }
    for (int i = 0; i < num_layers - 1; i++) {
        zs[i].resize(capacity * sizes[i + 1]);
        deltas[i].resize(capacity * sizes[i + 1]);
    }
}

void SigmoidMLP::upload_batch(const std::vector<std::pair<Vector, Vector>>& batch) {
    std::vector<float> x;
    std::vector<int> y;
    for (int i = 0; i < batch.size(); i++) {
        x.insert(x.end(), batch[i].first.begin(), batch[i].first.end());
        y.push_back(argmax(batch[i].second));
    }
    batch_x.upload(x);
    batch_y.upload(y);
}

const float* SigmoidMLP::output() const {
    return act_ptrs[num_layers - 1];
}

Vector SigmoidMLP::feed_forward(const Vector& a) {
    batch_x.upload(std::vector<float>(a.begin(), a.end()));
    this->forward(batch_x.data(), 1);

    std::vector<float> arr = activations[num_layers - 1].download(sizes[num_layers - 1]);
    return Vector(arr.begin(), arr.end());
}

void SigmoidMLP::forward(const float* x, int batch_size) {
    this->ensure_capacity(batch_size);
    act_ptrs[0] = x;

    for (int i = 0; i < num_layers - 1; i++) {
        int n_in = sizes[i];
        int n_out = sizes[i + 1];

        // z = a * w^T + b, then a = sigmoid(z)
        gemm(false, true, batch_size, n_out, n_in, 1.0f, act_ptrs[i], weights[i].data(), 0.0f, zs[i].data());
        bias_sigmoid_kernel<<<num_blocks(batch_size * n_out), BLOCK_SIZE>>>(zs[i].data(), biases[i].data(), batch_size, n_out, activations[i + 1].data());
        CUDA_CHECK_LAUNCH();

        act_ptrs[i + 1] = activations[i + 1].data();
    }
}

void SigmoidMLP::backpropagation(const float* x, const int* y, int batch_size, bool cross_entropy) {

    // forward pass

    this->forward(x, batch_size);

    // backward pass

    int last = num_layers - 2;
    output_delta_kernel<<<num_blocks(batch_size * sizes[last + 1]), BLOCK_SIZE>>>(cross_entropy, this->output(), y, batch_size, sizes[last + 1], deltas[last].data());
    CUDA_CHECK_LAUNCH();

    for (int i = last; i >= 0; i--) {
        int n_in = sizes[i];
        int n_out = sizes[i + 1];

        // hidden layers error

        if (i < last) {
            // transpose weight multiplication
            gemm(false, false, batch_size, n_out, sizes[i + 2], 1.0f, deltas[i + 1].data(), weights[i + 1].data(), 0.0f, deltas[i].data());

            // apply sigmoid prime
            sigmoid_prime_kernel<<<num_blocks(batch_size * n_out), BLOCK_SIZE>>>(deltas[i].data(), zs[i].data(), batch_size * n_out);
            CUDA_CHECK_LAUNCH();
        }

        // gradients summed over the batch
        column_sums(deltas[i].data(), batch_size, n_out, nabla_b[i].data());
        gemm(true, false, n_out, n_in, batch_size, 1.0f, deltas[i].data(), act_ptrs[i], 0.0f, nabla_w[i].data());
    }
}

void SigmoidMLP::apply_gradients(float factor, float decay) {
    for (int i = 0; i < num_layers - 1; i++) {
        sgd_step(weights[i].data(), nabla_w[i].data(), weights[i].size(), decay, factor);
        sgd_step(biases[i].data(), nabla_b[i].data(), biases[i].size(), 1.0f, factor);
    }
}

int SigmoidMLP::count_correct(const DeviceDataset& data) {
    DeviceBuffer<int> count(1);
    count.zero();

    for (int start = 0; start < data.n; start += EVAL_BATCH_SIZE) {
        int batch_size = std::min(EVAL_BATCH_SIZE, data.n - start);
        this->forward(data.image(start), batch_size);
        gpu::count_correct(this->output(), data.label(start), batch_size, sizes[num_layers - 1], count.data());
    }
    return count.download()[0];
}

std::vector<float> SigmoidMLP::sample_costs(const DeviceDataset& data, bool cross_entropy) {
    DeviceBuffer<float> costs(data.n);
    int last = num_layers - 2;

    for (int start = 0; start < data.n; start += EVAL_BATCH_SIZE) {
        int batch_size = std::min(EVAL_BATCH_SIZE, data.n - start);
        this->forward(data.image(start), batch_size);
        sample_cost_kernel<<<num_blocks(batch_size), BLOCK_SIZE>>>(cross_entropy, zs[last].data(), this->output(), data.label(start), batch_size, sizes[last + 1], costs.data() + start);
        CUDA_CHECK_LAUNCH();
    }
    return costs.download();
}

// network

Network::Network(const std::vector<int>& sizes) : SigmoidMLP(sizes) {
    // random number gen initialize
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);

    // initialize biases
    Matrix b;
    for (int i = 1; i < sizes.size(); i++) {
        Vector layer_biases;
        for (int j = 0; j < sizes[i]; j++) {
            layer_biases.push_back(dis(gen));
        }
        b.push_back(layer_biases);
    }

    // initialize weights
    Tensor w;
    for (int i = 0; i < sizes.size() - 1; i++) {
        Matrix layer_weights;
        int row = sizes[i + 1];
        int col = sizes[i];

        for (int j = 0; j < row; j++) {
            Vector neuron_weights;
            for (int k = 0; k < col; k++) {
                neuron_weights.push_back(dis(gen));
            }
            layer_weights.push_back(neuron_weights);
        }
        w.push_back(layer_weights);
    }

    this->set_parameters(b, w);
}

void Network::sgd(
    std::vector<std::pair<Vector, Vector>>& training_data,
    int epochs,
    int mini_batch_size,
    double learning_rate,
    const std::vector<std::pair<Vector, int>>* test_data)
{
    int n = training_data.size();

    // data is copied to the GPU once, then an index list is shuffled instead of the images
    DeviceDataset train = upload_dataset(training_data);
    DeviceDataset test;
    if (test_data != nullptr) {
        test = upload_dataset(*test_data);
    }

    std::vector<int> perm(n);
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    DeviceBuffer<int> perm_dev(n);

    batch_x.resize(mini_batch_size * train.dim);
    batch_y.resize(mini_batch_size);

    std::random_device rd;
    std::mt19937 g(rd());

    for (int i = 0; i < epochs; i++) {
        std::shuffle(perm.begin(), perm.end(), g);
        perm_dev.upload(perm);

        for (int j = 0; j < n; j += mini_batch_size) {
            int count = std::min(mini_batch_size, n - j);
            gather_batch(train, perm_dev.data(), j, count, batch_x.data(), batch_y.data());
            this->update_batch(batch_x.data(), batch_y.data(), count, learning_rate);
        }

        if (test_data != nullptr) {
            double curr_accuracy = (static_cast<double>(this->evaluate(test)) / test.n) * 100.0;
            double curr_cost = this->calculate_cost(train);
            std::cout << "Epoch " << i + 1 << " with Accuracy: " << curr_accuracy << "% and Cost: " << curr_cost << std::endl;
        } else {
            std::cout << "Epoch " << i + 1 << " complete" << std::endl;
        }
    }
}

void Network::update_mini_batch(std::vector<std::pair<Vector, Vector>>& mini_batch, double learning_rate) {
    this->upload_batch(mini_batch);
    this->update_batch(batch_x.data(), batch_y.data(), mini_batch.size(), learning_rate);
}

void Network::update_batch(const float* x, const int* y, int batch_size, double learning_rate) {
    this->backpropagation(x, y, batch_size, false);

    double factor = learning_rate / batch_size;
    this->apply_gradients(factor, 1.0f);
}

int Network::evaluate(const std::vector<std::pair<Vector, int>>& test_data) {
    return this->evaluate(upload_dataset(test_data));
}

int Network::evaluate(const DeviceDataset& test_data) {
    return this->count_correct(test_data);
}

double Network::calculate_cost(const std::vector<std::pair<Vector, Vector>>& data) {
    return this->calculate_cost(upload_dataset(data));
}

double Network::calculate_cost(const DeviceDataset& data) {
    std::vector<float> costs = this->sample_costs(data, false);

    double total_loss = 0.0;
    for (int i = 0; i < costs.size(); i++) {
        total_loss += costs[i];
    }
    return total_loss / data.n;
}

}
