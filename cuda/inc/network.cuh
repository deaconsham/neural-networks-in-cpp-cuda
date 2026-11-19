/*
Header file for SGD learning algorithm for a MNIST-trained neural network on GPU (CUDA).
Each mini-batch is one matrix with a row per image, instead of one image at a time.
Gradients are calculated using backpropagation.
*/

#ifndef CUDA_NETWORK_CUH
#define CUDA_NETWORK_CUH

#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

#include "common.h"
#include "common.cuh"

namespace gpu {

// sigmoid layers and the forward / backward passes, shared by Network and Network2
class SigmoidMLP {
    public:

    int num_layers;
    std::vector<int> sizes;
    std::vector<DeviceBuffer<float>> biases;
    std::vector<DeviceBuffer<float>> weights; // [sizes[l + 1]][sizes[l]], same as the CPU network

    // summed over the most recent batch
    std::vector<DeviceBuffer<float>> nabla_b;
    std::vector<DeviceBuffer<float>> nabla_w;

    SigmoidMLP(const std::vector<int>& sizes);

    Matrix get_biases() const;

    Tensor get_weights() const;

    void set_parameters(const Matrix& biases, const Tensor& weights);

    Vector feed_forward(const Vector& a);

    void forward(const float* x, int batch_size);

    void backpropagation(const float* x, const int* y, int batch_size, bool cross_entropy);

    void apply_gradients(float factor, float decay);

    const float* output() const;

    int count_correct(const DeviceDataset& data);

    // cost of each image, not averaged or regularized
    std::vector<float> sample_costs(const DeviceDataset& data, bool cross_entropy);

    protected:

    // workspace for one batch, grows when a larger batch comes in
    int capacity = 0;
    std::vector<const float*> act_ptrs; // act_ptrs[0] is the input
    std::vector<DeviceBuffer<float>> activations; // activations[0] is unused
    std::vector<DeviceBuffer<float>> zs;
    std::vector<DeviceBuffer<float>> deltas;

    DeviceBuffer<float> batch_x;
    DeviceBuffer<int> batch_y;

    void ensure_capacity(int batch_size);

    void upload_batch(const std::vector<std::pair<Vector, Vector>>& batch);
};

class Network : public SigmoidMLP {
    public:

    Network(const std::vector<int>& sizes);

    void sgd(std::vector<std::pair<Vector, Vector>>& training_data, int epochs, int mini_batch_size, double learning_rate, const std::vector<std::pair<Vector, int>>* test_data = nullptr);

    void update_mini_batch(std::vector<std::pair<Vector, Vector>>& mini_batch, double learning_rate);

    void update_batch(const float* x, const int* y, int batch_size, double learning_rate);

    int evaluate(const std::vector<std::pair<Vector, int>>& test_data);

    int evaluate(const DeviceDataset& test_data);

    double calculate_cost(const std::vector<std::pair<Vector, Vector>>& data);

    double calculate_cost(const DeviceDataset& data);
};

}

#endif
