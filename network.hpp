#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <utility>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Tensor = std::vector<Matrix>;

class Network {
    public:

    int num_layers;
    std::vector<int> sizes;

    Matrix biases;
    Tensor weights;

    Network(const std::vector<int>& sizes);

    Vector feed_forward(const Vector& a);

    void sgd(
        std::vector<std::pair<Vector, Vector>>& training_data, 
        int epochs, 
        int mini_batch_size, 
        double learning_rate, 
        const std::vector<std::pair<Vector, int>>* test_data = nullptr
    );

    void update_mini_batch(std::vector<std::pair<Vector, Vector>>& mini_batch, double learning_rate);
    
    std::pair<Matrix, Tensor> backpropagation(const Vector& x, const Vector& y);
    
    int evaluate(const std::vector<std::pair<Vector, int>>& test_data);
    
    Vector cost_derivative(const Vector& output_activation, const Vector& y);
};

Vector sigmoid(const Vector& z);

Vector sigmoid_prime(const Vector& z);

#endif