/*
Improved SGD learning algorithm from Chapter 3 (network2.py) on CPU.
Adds the cross-entropy cost, L2 regularization, better weight initialization,
monitoring of cost and accuracy, early stopping and saving / loading the network.
Gradients are calculated using backpropagation.
*/

#include "network2.h"

Network2::Network2(const std::vector<int>& sizes, Cost cost) {
    num_layers = sizes.size();
    this->sizes = sizes;
    this->cost = cost;

    std::random_device rd;
    gen.seed(rd());

    this->default_weight_initializer();
}

void Network2::default_weight_initializer() {
    // biases ~ N(0, 1), weights ~ N(0, 1/n_in) so hidden neurons don't start saturated
    std::normal_distribution<double> dis(0.0, 1.0);

    biases.clear();
    for (int i = 1; i < sizes.size(); i++) {
        Vector layer_biases(sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            layer_biases[j] = dis(gen);
        }
        biases.push_back(layer_biases);
    }

    weights.clear();
    for (int i = 0; i < sizes.size() - 1; i++) {
        Matrix layer_weights(sizes[i + 1], Vector(sizes[i]));
        double scale = 1.0 / std::sqrt(static_cast<double>(sizes[i]));
        for (int j = 0; j < sizes[i + 1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                layer_weights[j][k] = dis(gen) * scale;
            }
        }
        weights.push_back(layer_weights);
    }
}

void Network2::large_weight_initializer() {
    // same initialization as Chapter 1, weights and biases ~ N(0, 1)
    std::normal_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < biases.size(); i++) {
        for (int j = 0; j < biases[i].size(); j++) {
            biases[i][j] = dis(gen);
        }
    }

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] = dis(gen);
            }
        }
    }
}

Vector Network2::feed_forward(const Vector& a) {
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

Monitor Network2::sgd(
    std::vector<std::pair<Vector, Vector>>& training_data,
    int epochs,
    int mini_batch_size,
    double learning_rate,
    double lmbda,
    const std::vector<std::pair<Vector, int>>* evaluation_data,
    bool monitor_evaluation_cost,
    bool monitor_evaluation_accuracy,
    bool monitor_training_cost,
    bool monitor_training_accuracy,
    int early_stopping_n)
{
    int n = training_data.size();
    Monitor monitor;

    // early stopping
    int best_accuracy = 0;
    int no_accuracy_change = 0;

    for (int i = 0; i < epochs; i++) {
        std::shuffle(training_data.begin(), training_data.end(), gen);
        for (int j = 0; j < n; j += mini_batch_size) {
            int end = std::min(j + mini_batch_size, n);

            std::vector<std::pair<Vector, Vector>> mini_batch(
                training_data.begin() + j,
                training_data.begin() + end
            );
            this->update_mini_batch(mini_batch, learning_rate, lmbda, n);
        }

        std::cout << "Epoch " << i + 1 << " training complete" << std::endl;

        if (monitor_training_cost) {
            double c = this->total_cost(training_data, lmbda);
            monitor.training_cost.push_back(c);
            std::cout << "Cost on training data: " << c << std::endl;
        }
        if (monitor_training_accuracy) {
            int a = this->accuracy(training_data);
            monitor.training_accuracy.push_back(a);
            std::cout << "Accuracy on training data: " << a << " / " << n << std::endl;
        }
        if (evaluation_data != nullptr && monitor_evaluation_cost) {
            double c = this->total_cost(*evaluation_data, lmbda);
            monitor.evaluation_cost.push_back(c);
            std::cout << "Cost on evaluation data: " << c << std::endl;
        }
        if (evaluation_data != nullptr && monitor_evaluation_accuracy) {
            int a = this->accuracy(*evaluation_data);
            monitor.evaluation_accuracy.push_back(a);
            std::cout << "Accuracy on evaluation data: " << a << " / " << evaluation_data->size() << std::endl;
        }

        // stop if evaluation accuracy hasn't improved in the last early_stopping_n epochs
        if (early_stopping_n > 0 && !monitor.evaluation_accuracy.empty()) {
            int curr_accuracy = monitor.evaluation_accuracy.back();
            if (curr_accuracy > best_accuracy) {
                best_accuracy = curr_accuracy;
                no_accuracy_change = 0;
            } else {
                no_accuracy_change++;
            }

            if (no_accuracy_change == early_stopping_n) {
                std::cout << "Early-stopping: No accuracy change in last epochs: " << early_stopping_n << std::endl;
                return monitor;
            }
        }
        std::cout << std::endl;
    }
    return monitor;
}

void Network2::update_mini_batch(std::vector<std::pair<Vector, Vector>>& mini_batch, double learning_rate, double lmbda, int n) {
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
        std::pair<Matrix, Tensor> deltas = this->backpropagation(mini_batch[i].first, mini_batch[i].second);

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

    // L2 regularization shrinks the weights by a constant factor each step
    double decay = 1.0 - learning_rate * (lmbda / n);

    for (int i = 0; i < biases.size(); i++) {
        for (int j = 0; j < biases[i].size(); j++) {
            biases[i][j] -= factor * nabla_b[i][j];
        }
    }

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                weights[i][j][k] = decay * weights[i][j][k] - factor * nabla_w[i][j][k];
            }
        }
    }
}

std::pair<Matrix, Tensor> Network2::backpropagation(const Vector& x, const Vector& y) {
    Matrix nabla_b = biases;
    Tensor nabla_w = weights;

    Vector activation = x;
    Matrix activations;
    activations.push_back(activation);
    Matrix zs;

    // forward pass

    for (int i = 0; i < biases.size(); i++) {
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

    // backward pass, output error depends on the cost function

    Vector delta = this->cost_delta(zs[zs.size() - 1], activations[activations.size() - 1], y);

    for (int l = 1; l < num_layers; l++) {
        int idx = biases.size() - l;

        if (l > 1) {
            // transpose weight multiplication, then apply sigmoid prime
            Vector sp = sigmoid_prime(zs[idx]);
            Vector new_delta(weights[idx + 1][0].size(), 0.0);
            for (int j = 0; j < weights[idx + 1].size(); j++) {
                for (int k = 0; k < weights[idx + 1][j].size(); k++) {
                    new_delta[k] += weights[idx + 1][j][k] * delta[j];
                }
            }
            for (int j = 0; j < new_delta.size(); j++) {
                new_delta[j] *= sp[j];
            }
            delta = new_delta;
        }

        nabla_b[idx] = delta;

        const Vector& prev_activation = activations[idx];
        for (int j = 0; j < delta.size(); j++) {
            for (int k = 0; k < prev_activation.size(); k++) {
                nabla_w[idx][j][k] = delta[j] * prev_activation[k];
            }
        }
    }
    return std::pair<Matrix, Tensor>(nabla_b, nabla_w);
}

int Network2::accuracy(const std::vector<std::pair<Vector, Vector>>& data) {
    int count = 0;
    for (int i = 0; i < data.size(); i++) {
        if (argmax(this->feed_forward(data[i].first)) == argmax(data[i].second)) {
            count++;
        }
    }
    return count;
}

int Network2::accuracy(const std::vector<std::pair<Vector, int>>& data) {
    int count = 0;
    for (int i = 0; i < data.size(); i++) {
        if (argmax(this->feed_forward(data[i].first)) == data[i].second) {
            count++;
        }
    }
    return count;
}

double Network2::total_cost(const std::vector<std::pair<Vector, Vector>>& data, double lmbda) {
    double c = 0.0;
    for (int i = 0; i < data.size(); i++) {
        Vector a = this->feed_forward(data[i].first);
        c += this->cost_fn(a, data[i].second) / data.size();
    }

    // L2 regularization term
    double sum_sq = 0.0;
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                sum_sq += weights[i][j][k] * weights[i][j][k];
            }
        }
    }
    c += 0.5 * (lmbda / data.size()) * sum_sq;
    return c;
}

double Network2::total_cost(const std::vector<std::pair<Vector, int>>& data, double lmbda) {
    std::vector<std::pair<Vector, Vector>> converted;
    converted.reserve(data.size());
    for (int i = 0; i < data.size(); i++) {
        converted.push_back(std::make_pair(data[i].first, vectorized_result(data[i].second)));
    }
    return this->total_cost(converted, lmbda);
}

double Network2::cost_fn(const Vector& a, const Vector& y) {
    double c = 0.0;
    if (cost == Cost::QUADRATIC) {
        // 0.5 * ||a - y||^2
        for (int i = 0; i < a.size(); i++) {
            c += 0.5 * (a[i] - y[i]) * (a[i] - y[i]);
        }
    } else {
        // -y ln(a) - (1 - y) ln(1 - a), skipping terms that would be 0 * log(0)
        for (int i = 0; i < a.size(); i++) {
            if (y[i] > 0.0) {
                c -= y[i] * std::log(a[i]);
            }
            if (y[i] < 1.0) {
                c -= (1.0 - y[i]) * std::log(1.0 - a[i]);
            }
        }
        if (std::isnan(c) || std::isinf(c)) {
            c = std::numeric_limits<double>::max();
        }
    }
    return c;
}

Vector Network2::cost_delta(const Vector& z, const Vector& a, const Vector& y) {
    Vector delta(a.size());
    if (cost == Cost::QUADRATIC) {
        // (a - y) * sigmoid'(z)
        Vector sp = sigmoid_prime(z);
        for (int i = 0; i < a.size(); i++) {
            delta[i] = (a[i] - y[i]) * sp[i];
        }
    } else {
        // the sigmoid'(z) term cancels for cross-entropy, so no learning slowdown
        for (int i = 0; i < a.size(); i++) {
            delta[i] = a[i] - y[i];
        }
    }
    return delta;
}

Matrix Network2::average_gradient(const std::vector<std::pair<Vector, Vector>>& data) {
    // average of dC/db over data for every layer, used in Chapter 5
    Matrix gradient = biases;
    for (int i = 0; i < gradient.size(); i++) {
        std::fill(gradient[i].begin(), gradient[i].end(), 0.0);
    }

    for (int i = 0; i < data.size(); i++) {
        Matrix nabla_b = this->backpropagation(data[i].first, data[i].second).first;
        for (int j = 0; j < gradient.size(); j++) {
            for (int k = 0; k < gradient[j].size(); k++) {
                gradient[j][k] += nabla_b[j][k] / data.size();
            }
        }
    }
    return gradient;
}

void Network2::save(const std::string& filename) {
    // plain text: sizes, cost, then biases and weights layer by layer
    std::ofstream ofs(filename);
    assert(ofs.is_open() && "Cannot open file to save network");

    ofs << std::setprecision(17);
    ofs << sizes.size();
    for (int i = 0; i < sizes.size(); i++) {
        ofs << " " << sizes[i];
    }
    ofs << "\n" << (cost == Cost::QUADRATIC ? "QuadraticCost" : "CrossEntropyCost") << "\n";

    for (int i = 0; i < biases.size(); i++) {
        for (int j = 0; j < biases[i].size(); j++) {
            ofs << biases[i][j] << " ";
        }
        ofs << "\n";
    }

    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                ofs << weights[i][j][k] << " ";
            }
        }
        ofs << "\n";
    }
    ofs.close();
}

Network2 load(const std::string& filename) {
    std::ifstream ifs(filename);
    assert(ifs.is_open() && "Cannot open file to load network");

    int num_layers;
    ifs >> num_layers;
    std::vector<int> sizes(num_layers);
    for (int i = 0; i < num_layers; i++) {
        ifs >> sizes[i];
    }

    std::string cost_name;
    ifs >> cost_name;
    Cost cost = (cost_name == "QuadraticCost") ? Cost::QUADRATIC : Cost::CROSS_ENTROPY;

    Network2 net(sizes, cost);

    for (int i = 0; i < net.biases.size(); i++) {
        for (int j = 0; j < net.biases[i].size(); j++) {
            ifs >> net.biases[i][j];
        }
    }

    for (int i = 0; i < net.weights.size(); i++) {
        for (int j = 0; j < net.weights[i].size(); j++) {
            for (int k = 0; k < net.weights[i][j].size(); k++) {
                ifs >> net.weights[i][j][k];
            }
        }
    }
    ifs.close();
    return net;
}

