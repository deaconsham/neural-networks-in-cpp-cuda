/*
Header file for the improved SGD learning algorithm from Chapter 3 (network2.py) on CPU.
Adds the cross-entropy cost, L2 regularization, better weight initialization,
monitoring of cost and accuracy, early stopping and saving / loading the network.
Gradients are calculated using backpropagation.
*/

#ifndef NETWORK2_H
#define NETWORK2_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>

#include "common.h"
#include "network.h" // reuses sigmoid and sigmoid_prime

enum class Cost {
    QUADRATIC,
    CROSS_ENTROPY
};

// results recorded each epoch by sgd when monitoring is turned on
struct Monitor {
    std::vector<double> evaluation_cost;
    std::vector<int> evaluation_accuracy;
    std::vector<double> training_cost;
    std::vector<int> training_accuracy;
};

class Network2 {
    public:

    int num_layers;
    std::vector<int> sizes;
    Matrix biases;
    Tensor weights;
    Cost cost;
    std::mt19937 gen;

    Network2(const std::vector<int>& sizes, Cost cost = Cost::CROSS_ENTROPY);

    void default_weight_initializer();

    void large_weight_initializer();

    Vector feed_forward(const Vector& a);

    Monitor sgd(
        std::vector<std::pair<Vector, Vector>>& training_data,
        int epochs,
        int mini_batch_size,
        double learning_rate,
        double lmbda = 0.0,
        const std::vector<std::pair<Vector, int>>* evaluation_data = nullptr,
        bool monitor_evaluation_cost = false,
        bool monitor_evaluation_accuracy = false,
        bool monitor_training_cost = false,
        bool monitor_training_accuracy = false,
        int early_stopping_n = 0);

    void update_mini_batch(std::vector<std::pair<Vector, Vector>>& mini_batch, double learning_rate, double lmbda, int n);

    std::pair<Matrix, Tensor> backpropagation(const Vector& x, const Vector& y);

    int accuracy(const std::vector<std::pair<Vector, Vector>>& data);

    int accuracy(const std::vector<std::pair<Vector, int>>& data);

    double total_cost(const std::vector<std::pair<Vector, Vector>>& data, double lmbda);

    double total_cost(const std::vector<std::pair<Vector, int>>& data, double lmbda);

    double cost_fn(const Vector& a, const Vector& y);

    Vector cost_delta(const Vector& z, const Vector& a, const Vector& y);

    Matrix average_gradient(const std::vector<std::pair<Vector, Vector>>& data);

    void save(const std::string& filename);
};

Network2 load(const std::string& filename);

#endif
