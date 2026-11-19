/*
SGD learning algorithm for a MNIST-trained neural network on CPU.
Gradients are calculated using backpropagation.
*/

#include "network.h"

Network::Network(const std::vector<int>& sizes) {
    num_layers = sizes.size();

    // random number gen initialize
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);

    // initialize biases
    for (int i = 1; i < sizes.size(); i++) {
        Vector layer_biases;
        for (int j = 0; j < sizes[i]; j++) {
            layer_biases.push_back(dis(gen));
        }
        biases.push_back(layer_biases);
    }

    // initialize weights
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
        weights.push_back(layer_weights);
    }
}

Vector Network::feed_forward(const Vector& a) {
    Vector arr = a;
    for (int i = 0; i < weights.size(); i++) {
        Vector z_layer(weights[i].size()); 
        for (int j = 0; j < weights[i].size(); j++) {
            double sum = 0.0;
            for (int k = 0; k < arr.size(); k++) {
                sum += weights[i][j][k] * arr[k]; 
            }
            z_layer[j] = sum + biases[i][j];
        }
        arr = sigmoid(z_layer);
    }
    return arr;
}

void Network::sgd(
    std::vector<std::pair<Vector, Vector>>& training_data, 
    int epochs, 
    int mini_batch_size, 
    double learning_rate, 
    const std::vector<std::pair<Vector, int>>* test_data)
{
    int n = training_data.size();

    std::random_device rd;
    std::mt19937 g(rd());

    for (int i = 0; i < epochs; i++) {
        std::shuffle(training_data.begin(), training_data.end(), g);
        for (int j = 0; j < n; j+= mini_batch_size) {
            int end = std::min(j + mini_batch_size, n);
            
            std::vector<std::pair<Vector, Vector>> mini_batch(
                training_data.begin() + j,
                training_data.begin() + end
            );
            this->update_mini_batch(mini_batch, learning_rate);
        }
        
        if (test_data != nullptr) {
            double curr_accuracy = (static_cast<double>(this->evaluate(*test_data)) / test_data->size()) * 100.0;
            double curr_cost = this->calculate_cost(training_data);
            std::cout << "Epoch " << i + 1 << " with Accuracy: " << curr_accuracy << "% and Cost: " << curr_cost << std::endl;
        } else {
            std::cout << "Epoch " << i + 1 << " complete" << std::endl;
        }
    }
}

void Network::update_mini_batch(std::vector<std::pair<Vector, Vector>>& mini_batch, double learning_rate) {
    Matrix nabla_b = biases;
    for (int i = 0; i < nabla_b.size(); i++) {
        std::fill(nabla_b[i].begin(), nabla_b[i].end(), 0.0);
    }

    Tensor nabla_w = weights;
    for (int i = 0; i < nabla_w.size(); i++) {
        for (int j = 0; j < nabla_w[i].size(); j++) {
            std::fill(nabla_w[i][j].begin(), nabla_w[i][j].end(), 0.0);
        }
    }

    for (int i = 0; i < mini_batch.size(); i++) {
        const Vector& x = mini_batch[i].first;
        const Vector& y = mini_batch[i].second;

        std::pair<Matrix, Tensor> deltas = this->backpropagation(x, y);

        Matrix& delta_nabla_b = deltas.first;
        Tensor& delta_nabla_w = deltas.second;

        for (int j = 0; j < nabla_b.size(); j++) {
            for (int k = 0; k < nabla_b[j].size(); k++) {
                nabla_b[j][k] += delta_nabla_b[j][k];
            }
        }

        for (int j = 0; j < nabla_w.size(); j++) {
            for (int k = 0; k < nabla_w[j].size(); k++) {
                for (int l = 0; l < nabla_w[j][k].size(); l++) {
                    nabla_w[j][k][l] += delta_nabla_w[j][k][l];
                }
            }
        }
    }

    double factor = learning_rate / mini_batch.size();

    for (int i = 0; i < biases.size(); i++) {
        for (int j = 0; j < biases[i].size(); j++) {
            biases[i][j] -= factor * nabla_b[i][j];
        }
    }

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] -= factor * nabla_w[i][j][k];
            }
        }
    }
}

std::pair<Matrix, Tensor> Network::backpropagation(const Vector& x, const Vector& y) {
    // copied only for their shape, every entry is overwritten below
    Matrix nabla_b = biases;
    Tensor nabla_w = weights;

    Vector activation = x;
    Matrix activations;
    activations.push_back(activation);
    Matrix zs;

    // forward pass

    for(int i = 0; i < biases.size(); i++) {
        Vector z_layer(biases[i].size());

        for (int j = 0; j < biases[i].size(); j++) {
            double sum = 0.0;
            for (int k = 0; k < activation.size(); k++) {
                sum += weights[i][j][k] * activation[k];
            }
            z_layer[j] = sum + biases[i][j];
        }

        zs.push_back(z_layer);
        activation = sigmoid(z_layer);
        activations.push_back(activation);
    }

    // backward pass

    int last_idx = zs.size() - 1;
    Vector delta = this->cost_derivative(activations[activations.size() - 1], y);
    Vector sp = sigmoid_prime(zs[last_idx]);

    for (int i = 0; i < delta.size(); i++) {
        delta[i] = delta[i] * sp[i];
    }

    nabla_b[nabla_b.size() - 1] = delta;

    int act_idx = activations.size() - 2;
    int nw_idx = nabla_w.size() - 1;

    for (int i = 0; i < delta.size(); i++) {
        for (int j = 0; j < activations[act_idx].size(); j++) {
            nabla_w[nw_idx][i][j] = delta[i] * activations[act_idx][j];
        }
    }


    // hidden layers error

    for (int i = 2; i < num_layers; i++) {
        Vector z = zs[zs.size() - i];
        Vector sp_hidden = sigmoid_prime(z);
        
        // transpose weight multiplication

        int w_idx = weights.size() - i + 1;
        Vector new_delta(weights[w_idx][0].size(), 0.0);
        for (int j = 0; j < weights[w_idx].size(); j++) {
            for (int k = 0; k < weights[w_idx][j].size(); k++) {
                new_delta[k] += weights[w_idx][j][k] * delta[j];
            }
        }

        // apply sigmoid prime

        delta = new_delta;
        for (int j = 0; j < delta.size(); j++) {
            delta[j] = delta[j] * sp_hidden[j];
        }

        nabla_b[nabla_b.size() - i] = delta;

        //weights gradient for hidden layer

        int prev_act_idx = activations.size() - i - 1;
        int curr_nw_idx = nabla_w.size() - i;

        for (int j = 0; j < delta.size(); j++) {
            for (int k = 0; k < activations[prev_act_idx].size(); k++) {
                nabla_w[curr_nw_idx][j][k] = delta[j] * activations[prev_act_idx][k];
            }
        }
    }
    return std::pair<Matrix, Tensor>(nabla_b, nabla_w);
}

int Network::evaluate(const std::vector<std::pair<Vector, int>>& test_data) {
    int count = 0;

    for (int i = 0; i < test_data.size(); i++) {
        const Vector& x = test_data[i].first;
        int y = test_data[i].second;

        // if neural network's guess matches the label, increment count
        if (argmax(this->feed_forward(x)) == y) {
            count++;
        }
    }
    return count;
}

Vector Network::cost_derivative(const Vector& output_activation, const Vector& y) {
    Vector arr(output_activation.size());
    for (int i = 0; i < output_activation.size(); i++) {
        arr[i] = output_activation[i] - y[i];
    }
    return arr;
}

double Network::calculate_cost(const std::vector<std::pair<Vector, Vector>>& data) {
    double total_loss = 0.0;
    for (int i = 0; i < data.size(); i++) {
        const Vector& x = data[i].first;
        const Vector& y = data[i].second;
        
        Vector prediction = this->feed_forward(x);
        
        double image_loss = 0.0;
        for (int j = 0; j < prediction.size(); j++) {
            double error = prediction[j] - y[j];
            image_loss += (error * error); 
        }
        total_loss += (0.5 * image_loss);
    }

    return total_loss / data.size();
}

Vector sigmoid(const Vector& z) {
    Vector arr(z.size());
    for (int i = 0; i < z.size(); i++) {
        arr[i] = 1.0 / (1.0 + std::exp(-z[i]));
    }
    return arr;
}

Vector sigmoid_prime(const Vector& z) {
    Vector arr(z.size());
    Vector sig = sigmoid(z);
    for (int i = 0; i < z.size(); i++) {
        arr[i] = sig[i] * (1.0 - sig[i]);
    }
    return arr;
}