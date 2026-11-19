/*
Header file for the final convolutional neural network from Chapter 6 (network3.py) on CPU.
Architecture: convolutional-pooling layers, fully-connected layers and a softmax output layer.
Uses ReLU activations, dropout, L2 regularization and the log-likelihood cost.
Gradients are calculated using backpropagation.
*/

#ifndef CONV_NETWORK_H
#define CONV_NETWORK_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>

#include "common.h"

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

    Tensor4D weights; // [filter][channel][row][col]
    Vector biases; // [filter]

    Tensor4D nabla_w;
    Vector nabla_b;

    // cache from the most recent forward pass, used by backpropagation
    Tensor input;
    Tensor z;
    IndexTensor max_idx;

    // image_shape = {channels, rows, cols}, filter_shape = {filters, channels, rows, cols}
    ConvPoolLayer(const std::vector<int>& image_shape, const std::vector<int>& filter_shape, int pool_size, std::mt19937& gen);

    Tensor feed_forward(const Tensor& x);

    Tensor backpropagation(const Tensor& delta);

    void zero_gradients();

    void update(double learning_rate, int mini_batch_size, double weight_decay);
};

class FullyConnectedLayer {
    public:

    int n_in;
    int n_out;
    double p_dropout;

    Matrix weights; // [n_out][n_in]
    Vector biases; // [n_out]

    Matrix nabla_w;
    Vector nabla_b;

    // cache from the most recent dropout forward pass, used by backpropagation
    Vector input;
    Vector mask;
    Vector z;

    // weights and biases start at zero
    FullyConnectedLayer(int n_in, int n_out, double p_dropout = 0.0);

    // weights ~ N(0, 1/n_out), biases ~ N(0, 1)
    FullyConnectedLayer(int n_in, int n_out, std::mt19937& gen, double p_dropout = 0.0);

    Vector feed_forward(const Vector& x);

    Vector feed_forward_dropout(const Vector& x, std::mt19937& gen);

    Vector backpropagation(const Vector& delta);

    void zero_gradients();

    void update(double learning_rate, int mini_batch_size, double weight_decay);

    // shared with SoftmaxLayer, which only changes the activation and the output error

    Vector weighted_input(const Vector& x, double scale);

    Vector weighted_input_dropout(const Vector& x, std::mt19937& gen);

    Vector backpropagate_dz(const Vector& dz);
};

class SoftmaxLayer : public FullyConnectedLayer {
    public:

    Vector output;

    SoftmaxLayer(int n_in, int n_out, double p_dropout = 0.0);

    Vector feed_forward(const Vector& x);

    Vector feed_forward_dropout(const Vector& x, std::mt19937& gen);

    Vector backpropagation(int y);
};

class ConvNetwork {
    public:

    std::vector<ConvPoolLayer> conv_layers;
    std::vector<FullyConnectedLayer> fc_layers;
    SoftmaxLayer softmax_layer;
    std::mt19937 gen;

    ConvNetwork(const std::vector<ConvPoolLayer>& conv_layers, const std::vector<FullyConnectedLayer>& fc_layers, const SoftmaxLayer& softmax_layer);

    Vector conv_feed_forward(const Vector& x);

    Vector feed_forward(const Vector& x);

    void sgd(std::vector<std::pair<Vector, int>>& training_data, int epochs, int mini_batch_size, double learning_rate, const std::vector<std::pair<Vector, int>>& validation_data, const std::vector<std::pair<Vector, int>>& test_data, double lmbda = 0.0);

    void update_mini_batch(std::vector<std::pair<Vector, int>>& mini_batch, double learning_rate, double lmbda, int num_training_batches);

    void backpropagation(const Vector& x, int y);

    double accuracy(const std::vector<std::pair<Vector, int>>& data);
};

double relu(double z);

double relu_prime(double z);

Vector softmax(const Vector& z);

Tensor to_image(const Vector& x, int channels, int rows, int cols);

Vector flatten(const Tensor& t);

#endif
