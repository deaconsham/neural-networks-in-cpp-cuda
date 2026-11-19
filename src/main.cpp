#include <iostream>
#include <string>

#include "network.h"
#include "network2.h"
#include "conv_network.h"
#include "average_darkness.h"
#include "expand_mnist.h"
#include "mnist_loader.h"

// split the 60,000 training images into 50,000 training and 10,000 validation, like mnist.pkl.gz
const int NUM_TRAINING = 50000;

// (image, digit) pairs for images [start, end)
std::vector<std::pair<Vector, int>> labelled_data(const MNIST_Loader& loader, int start, int end) {
    std::vector<std::pair<Vector, int>> data;
    data.reserve(end - start);
    for (int i = start; i < end; i++) {
        data.push_back(std::make_pair(loader.images[i], loader.labels[i]));
    }
    return data;
}

// (image, one-hot label) pairs for images [start, end), used as training data by network and network2
std::vector<std::pair<Vector, Vector>> vectorized_data(const MNIST_Loader& loader, int start, int end) {
    std::vector<std::pair<Vector, Vector>> data;
    data.reserve(end - start);
    for (int i = start; i < end; i++) {
        data.push_back(std::make_pair(loader.images[i], vectorized_result(loader.labels[i])));
    }
    return data;
}

// Chapter 1: naive baseline that guesses digits by how dark the image is (mnist_average_darkness.py)
void run_average_darkness(MNIST_Loader& train_loader, MNIST_Loader& test_loader) {
    std::vector<std::pair<Vector, int>> training_data = labelled_data(train_loader, 0, NUM_TRAINING);

    Vector avgs = avg_darknesses(training_data);

    int count = 0;
    for (int i = 0; i < test_loader.size; i++) {
        if (guess_digit(test_loader.images[i], avgs) == test_loader.labels[i]) {
            count++;
        }
    }
    std::cout << "Baseline classifier using average darkness of image." << std::endl;
    std::cout << count << " of " << test_loader.size << " values correct." << std::endl;
}

// Chapter 1 and 2: fully-connected network trained with SGD and backpropagation (network.py)
void run_network(MNIST_Loader& train_loader, MNIST_Loader& test_loader) {
    std::vector<std::pair<Vector, Vector>> training_data = vectorized_data(train_loader, 0, train_loader.size);
    std::vector<std::pair<Vector, int>> testing_data = labelled_data(test_loader, 0, test_loader.size);

    Network net({784, 30, 10});

    int epochs = 30;
    int mini_batch_size = 10;
    double learning_rate = 3.0;

    net.sgd(training_data, epochs, mini_batch_size, learning_rate, &testing_data);
}

// Chapter 3: cross-entropy, L2 regularization and better weight initialization (network2.py)
void run_network2(MNIST_Loader& train_loader, MNIST_Loader& test_loader) {
    std::vector<std::pair<Vector, Vector>> training_data = vectorized_data(train_loader, 0, NUM_TRAINING);
    std::vector<std::pair<Vector, int>> validation_data = labelled_data(train_loader, NUM_TRAINING, train_loader.size);
    std::vector<std::pair<Vector, int>> testing_data = labelled_data(test_loader, 0, test_loader.size);

    Network2 net({784, 100, 10}, Cost::CROSS_ENTROPY);

    int epochs = 30;
    int mini_batch_size = 10;
    double learning_rate = 0.5;
    double lmbda = 5.0;

    net.sgd(training_data, epochs, mini_batch_size, learning_rate, lmbda, &validation_data,
            true, true, true, true);

    std::cout << "Accuracy on test data: " << net.accuracy(testing_data) << " / " << testing_data.size() << std::endl;

    net.save("network2.txt");
    std::cout << "Saved network to network2.txt" << std::endl;
}

// Chapter 5: vanishing gradient, how fast each hidden layer learns (generate_gradient.py)
void run_vanishing_gradient(MNIST_Loader& train_loader) {
    std::vector<std::pair<Vector, Vector>> training_data = vectorized_data(train_loader, 0, 1000);

    int epochs = 500;
    double learning_rate = 0.1;
    double lmbda = 5.0;

    for (int num_hidden = 2; num_hidden <= 4; num_hidden++) {
        std::vector<int> sizes = {784};
        for (int i = 0; i < num_hidden; i++) {
            sizes.push_back(30);
        }
        sizes.push_back(10);

        Network2 net(sizes, Cost::CROSS_ENTROPY);
        std::cout << "Network with " << num_hidden << " hidden layers of 30 neurons" << std::endl;

        for (int i = 0; i <= epochs; i++) {
            if (i % 100 == 0) {
                Matrix gradient = net.average_gradient(training_data);

                // length of dC/db for each hidden layer, the speed that layer learns
                std::cout << "Epoch " << i << ":";
                for (int l = 0; l < num_hidden; l++) {
                    double sum_sq = 0.0;
                    for (int j = 0; j < gradient[l].size(); j++) {
                        sum_sq += gradient[l][j] * gradient[l][j];
                    }
                    std::cout << "  layer " << l + 1 << " = " << std::sqrt(sum_sq);
                }
                std::cout << std::endl;
            }

            // one full-batch gradient descent step on the 1,000 images
            if (i < epochs) {
                net.update_mini_batch(training_data, learning_rate, lmbda, training_data.size());
            }
        }
        std::cout << std::endl;
    }
}

// Chapter 6: final convolutional network (network3.py)
void run_conv_network(MNIST_Loader& train_loader, MNIST_Loader& test_loader) {
    std::vector<std::pair<Vector, int>> training_data = labelled_data(train_loader, 0, NUM_TRAINING);
    std::vector<std::pair<Vector, int>> validation_data = labelled_data(train_loader, NUM_TRAINING, train_loader.size);
    std::vector<std::pair<Vector, int>> testing_data = labelled_data(test_loader, 0, test_loader.size);

    std::vector<std::pair<Vector, int>> expanded_training_data = expand_data(training_data, 28, 28);
    std::cout << "Expanded training data to " << expanded_training_data.size() << " images" << std::endl;

    int mini_batch_size = 10;
    int epochs = 40;
    double learning_rate = 0.03;
    double lmbda = 0.1;
    double p_dropout = 0.5;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<ConvPoolLayer> conv_layers = {
        ConvPoolLayer({1, 28, 28}, {20, 1, 5, 5}, 2, gen),      // 1x28x28 -> 20x24x24 -> 20x12x12
        ConvPoolLayer({20, 12, 12}, {40, 20, 5, 5}, 2, gen)     // 20x12x12 -> 40x8x8 -> 40x4x4
    };

    std::vector<FullyConnectedLayer> fc_layers = {
        FullyConnectedLayer(40 * 4 * 4, 1000, gen, p_dropout),
        FullyConnectedLayer(1000, 1000, gen, p_dropout)
    };

    SoftmaxLayer softmax_layer(1000, 10, p_dropout);

    ConvNetwork net(conv_layers, fc_layers, softmax_layer);

    net.sgd(expanded_training_data, epochs, mini_batch_size, learning_rate, validation_data, testing_data, lmbda);
}

int main(int argc, char* argv[]) {
    std::string mode = "ch6";
    if (argc > 1) {
        mode = argv[1];
    }

    MNIST_Loader train_loader("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");

    MNIST_Loader test_loader("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");

    if (mode == "darkness") {
        run_average_darkness(train_loader, test_loader);
    } else if (mode == "ch1") {
        run_network(train_loader, test_loader);
    } else if (mode == "ch3") {
        run_network2(train_loader, test_loader);
    } else if (mode == "ch5") {
        run_vanishing_gradient(train_loader);
    } else if (mode == "ch6") {
        run_conv_network(train_loader, test_loader);
    } else {
        std::cout << "Usage: ./neural_network [darkness | ch1 | ch3 | ch5 | ch6]" << std::endl;
        return 1;
    }

    return 0;
}
