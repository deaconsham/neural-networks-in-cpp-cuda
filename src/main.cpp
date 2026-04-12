#include <iostream>
#include <string>

#include "network.h"
#include "mnist_loader.h"

int main() {
    MNIST_Loader train_loader("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
    
    MNIST_Loader test_loader("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");

    std::vector<std::pair<std::vector<double>, std::vector<double>>> training_data;
    training_data.reserve(train_loader.size);

    for (int i = 0; i < train_loader.size; ++i) {
        std::vector<double> label(10, 0.0);
        label[train_loader.labels[i]] = 1.0; 
        training_data.push_back(std::make_pair(train_loader.images[i], label));
    }

    std::vector<std::pair<std::vector<double>, int>> testing_data;
    testing_data.reserve(test_loader.size);

    for (int i = 0; i < test_loader.size; ++i) {
        testing_data.push_back(std::make_pair(test_loader.images[i], test_loader.labels[i]));
    }
    
    Network net({784, 30, 10});

    int epochs = 30;
    int mini_batch_size = 10;
    double learning_rate = 3.0;
    
    net.sgd(training_data, epochs, mini_batch_size, learning_rate, &testing_data);

    return 0;
}