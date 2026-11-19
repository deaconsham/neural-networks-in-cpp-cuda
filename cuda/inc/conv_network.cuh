/*
Header file for the final convolutional neural network from Chapter 6 (network3.py) on GPU (CUDA).
Architecture: convolutional-pooling layers, fully-connected layers and a softmax output layer.
Uses ReLU activations, dropout, L2 regularization and the log-likelihood cost.
Gradients are calculated using backpropagation.
*/

#ifndef CUDA_CONV_NETWORK_CUH
#define CUDA_CONV_NETWORK_CUH

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "common.h"
#include "common.cuh"

namespace gpu {

// layer dimensions, passed to the kernels
struct ConvShape {
    int in_channels;
    int in_rows;
    int in_cols;
    int num_filters;
    int filter_rows;
    int filter_cols;
    int pool_size;
    int conv_rows;
    int conv_cols;
    int out_rows;
    int out_cols;
};

class ConvPoolLayer {
    public:

    int num_filters;
    int in_channels;
    int in_rows;
    int in_cols;
    int filter_rows;
    int filter_cols;
    int pool_size;
    int conv_rows;
    int conv_cols;
    int out_rows;
    int out_cols;

    DeviceBuffer<float> weights; // [filter][channel][row][col]
    DeviceBuffer<float> biases; // [filter]

    DeviceBuffer<float> nabla_w;
    DeviceBuffer<float> nabla_b;

    // cache from the most recent forward pass, used by backpropagation
    const float* input = nullptr; // [batch][channel][row][col], owned by the previous layer
    DeviceBuffer<float> z; // [batch][filter][row][col]
    DeviceBuffer<int> max_idx;
    DeviceBuffer<float> output;

    DeviceBuffer<float> dz;
    DeviceBuffer<float> delta_input;

    // image_shape = {channels, rows, cols}, filter_shape = {filters, channels, rows, cols}
    ConvPoolLayer(const std::vector<int>& image_shape, const std::vector<int>& filter_shape, int pool_size, std::mt19937& gen);

    ConvShape shape() const;

    int input_size() const;

    int output_size() const;

    const float* feed_forward(const float* x, int batch_size);

    // the first layer doesn't need delta_input, its input is the image
    const float* backpropagation(const float* delta, int batch_size, bool compute_delta_input);

    void update(double learning_rate, int mini_batch_size, double weight_decay);

    private:

    int capacity = 0;

    void ensure_capacity(int batch_size);
};

class FullyConnectedLayer {
    public:

    int n_in;
    int n_out;
    double p_dropout;

    DeviceBuffer<float> weights; // [n_out][n_in]
    DeviceBuffer<float> biases; // [n_out]

    DeviceBuffer<float> nabla_w;
    DeviceBuffer<float> nabla_b;

    // cache from the most recent dropout forward pass, used by backpropagation
    DeviceBuffer<float> input;
    DeviceBuffer<float> mask;
    DeviceBuffer<float> z;
    DeviceBuffer<float> output;

    DeviceBuffer<float> dz;
    DeviceBuffer<float> delta_input;

    // weights and biases start at zero
    FullyConnectedLayer(int n_in, int n_out, double p_dropout = 0.0);

    // weights ~ N(0, 1/n_out), biases ~ N(0, 1)
    FullyConnectedLayer(int n_in, int n_out, std::mt19937& gen, double p_dropout = 0.0);

    const float* feed_forward(const float* x, int batch_size);

    const float* feed_forward_dropout(const float* x, int batch_size, unsigned long long seed);

    const float* backpropagation(const float* delta, int batch_size);

    void update(double learning_rate, int mini_batch_size, double weight_decay);

    // shared with SoftmaxLayer, which only changes the activation and the output error

    void weighted_input(const float* x, int batch_size, float scale);

    void weighted_input_dropout(const float* x, int batch_size, unsigned long long seed);

    const float* backpropagate_dz(int batch_size);

    protected:

    int capacity = 0;

    void ensure_capacity(int batch_size);
};

class SoftmaxLayer : public FullyConnectedLayer {
    public:

    SoftmaxLayer(int n_in, int n_out, double p_dropout = 0.0);

    const float* feed_forward(const float* x, int batch_size);

    const float* feed_forward_dropout(const float* x, int batch_size, unsigned long long seed);

    const float* backpropagation(const int* labels, int batch_size);
};

class ConvNetwork {
    public:

    std::vector<ConvPoolLayer> conv_layers;
    std::vector<FullyConnectedLayer> fc_layers;
    SoftmaxLayer softmax_layer;
    std::mt19937 gen;

    ConvNetwork(const std::vector<ConvPoolLayer>& conv_layers, const std::vector<FullyConnectedLayer>& fc_layers, const SoftmaxLayer& softmax_layer);

    const float* conv_feed_forward(const float* x, int batch_size);

    const float* feed_forward(const float* x, int batch_size);

    void sgd(std::vector<std::pair<Vector, int>>& training_data, int epochs, int mini_batch_size, double learning_rate, const std::vector<std::pair<Vector, int>>& validation_data, const std::vector<std::pair<Vector, int>>& test_data, double lmbda = 0.0);

    void update_mini_batch(const float* x, const int* y, int batch_size, double learning_rate, double lmbda, int num_training_batches);

    void backpropagation(const float* x, const int* y, int batch_size);

    double accuracy(const DeviceDataset& data);

    double accuracy(const std::vector<std::pair<Vector, int>>& data);

    private:

    // seed for a new dropout mask
    unsigned long long random_seed();
};

}

#endif
