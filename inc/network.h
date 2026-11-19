/*
Header file for SGD learning algorithm for a MNIST-trained neural network on CPU.
Gradients are calculated using backpropagation.
*/

#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>

#include "common.h"

class Network {
    public:

    int num_layers;
    std::vector<int> sizes;
    Matrix biases;
    Tensor weights;

    Network(const std::vector<int>& sizes);

    Vector feed_forward(const Vector& a);

    void sgd(std::vector<std::pair<Vector, Vector>>& training_data, int epochs, int mini_batch_size, double learning_rate, const std::vector<std::pair<Vector, int>>* test_data = nullptr);

    void update_mini_batch(std::vector<std::pair<Vector, Vector>>& mini_batch, double learning_rate);
    
    std::pair<Matrix, Tensor> backpropagation(const Vector& x, const Vector& y);
    
    int evaluate(const std::vector<std::pair<Vector, int>>& test_data);
    
    Vector cost_derivative(const Vector& output_activation, const Vector& y);

    double calculate_cost(const std::vector<std::pair<Vector, Vector>>& training_data);
};

Vector sigmoid(const Vector& z);

Vector sigmoid_prime(const Vector& z);

#endif