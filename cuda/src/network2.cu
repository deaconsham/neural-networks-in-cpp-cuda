/*
Improved SGD learning algorithm from Chapter 3 (network2.py) on GPU (CUDA).
Adds the cross-entropy cost, L2 regularization, better weight initialization,
monitoring of cost and accuracy, early stopping and saving / loading the network.
Gradients are calculated using backpropagation.
*/

#include "network2.cuh"

namespace gpu {

Network2::Network2(const std::vector<int>& sizes, Cost cost) : SigmoidMLP(sizes) {
    this->cost = cost;

    std::random_device rd;
    gen.seed(rd());

    this->default_weight_initializer();
}

void Network2::default_weight_initializer() {
    // biases ~ N(0, 1), weights ~ N(0, 1/n_in) so hidden neurons don't start saturated
    std::normal_distribution<double> dis(0.0, 1.0);

    Matrix b;
    Tensor w;
    for (int i = 0; i < num_layers - 1; i++) {
        Vector layer_biases(sizes[i + 1]);
        for (int j = 0; j < sizes[i + 1]; j++) {
            layer_biases[j] = dis(gen);
        }
        b.push_back(layer_biases);
    }
    for (int i = 0; i < num_layers - 1; i++) {
        Matrix layer_weights(sizes[i + 1], Vector(sizes[i]));
        double scale = 1.0 / std::sqrt(static_cast<double>(sizes[i]));
        for (int j = 0; j < sizes[i + 1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                layer_weights[j][k] = dis(gen) * scale;
            }
        }
        w.push_back(layer_weights);
    }
    this->set_parameters(b, w);
}

void Network2::large_weight_initializer() {
    // same initialization as Chapter 1, weights and biases ~ N(0, 1)
    std::normal_distribution<double> dis(0.0, 1.0);

    Matrix b;
    Tensor w;
    for (int i = 0; i < num_layers - 1; i++) {
        Vector layer_biases(sizes[i + 1]);
        for (int j = 0; j < sizes[i + 1]; j++) {
            layer_biases[j] = dis(gen);
        }
        b.push_back(layer_biases);

        Matrix layer_weights(sizes[i + 1], Vector(sizes[i]));
        for (int j = 0; j < sizes[i + 1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                layer_weights[j][k] = dis(gen);
            }
        }
        w.push_back(layer_weights);
    }
    this->set_parameters(b, w);
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

    // data is copied to the GPU once, then an index list is shuffled instead of the images
    DeviceDataset train = upload_dataset(training_data);
    DeviceDataset evaluation;
    if (evaluation_data != nullptr) {
        evaluation = upload_dataset(*evaluation_data);
    }

    std::vector<int> perm(n);
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    DeviceBuffer<int> perm_dev(n);

    batch_x.resize(mini_batch_size * train.dim);
    batch_y.resize(mini_batch_size);

    // early stopping
    int best_accuracy = 0;
    int no_accuracy_change = 0;

    for (int i = 0; i < epochs; i++) {
        std::shuffle(perm.begin(), perm.end(), gen);
        perm_dev.upload(perm);

        for (int j = 0; j < n; j += mini_batch_size) {
            int count = std::min(mini_batch_size, n - j);
            gather_batch(train, perm_dev.data(), j, count, batch_x.data(), batch_y.data());
            this->update_batch(batch_x.data(), batch_y.data(), count, learning_rate, lmbda, n);
        }

        std::cout << "Epoch " << i + 1 << " training complete" << std::endl;

        if (monitor_training_cost) {
            double c = this->total_cost(train, lmbda);
            monitor.training_cost.push_back(c);
            std::cout << "Cost on training data: " << c << std::endl;
        }
        if (monitor_training_accuracy) {
            int a = this->accuracy(train);
            monitor.training_accuracy.push_back(a);
            std::cout << "Accuracy on training data: " << a << " / " << n << std::endl;
        }
        if (evaluation_data != nullptr && monitor_evaluation_cost) {
            double c = this->total_cost(evaluation, lmbda);
            monitor.evaluation_cost.push_back(c);
            std::cout << "Cost on evaluation data: " << c << std::endl;
        }
        if (evaluation_data != nullptr && monitor_evaluation_accuracy) {
            int a = this->accuracy(evaluation);
            monitor.evaluation_accuracy.push_back(a);
            std::cout << "Accuracy on evaluation data: " << a << " / " << evaluation.n << std::endl;
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
    this->upload_batch(mini_batch);
    this->update_batch(batch_x.data(), batch_y.data(), mini_batch.size(), learning_rate, lmbda, n);
}

void Network2::update_batch(const float* x, const int* y, int batch_size, double learning_rate, double lmbda, int n) {
    this->backpropagation(x, y, batch_size, cost == Cost::CROSS_ENTROPY);

    // L2 regularization shrinks the weights by a constant factor each step
    double factor = learning_rate / batch_size;
    double decay = 1.0 - learning_rate * (lmbda / n);
    this->apply_gradients(factor, decay);
}

int Network2::accuracy(const std::vector<std::pair<Vector, Vector>>& data) {
    return this->accuracy(upload_dataset(data));
}

int Network2::accuracy(const std::vector<std::pair<Vector, int>>& data) {
    return this->accuracy(upload_dataset(data));
}

int Network2::accuracy(const DeviceDataset& data) {
    return this->count_correct(data);
}

double Network2::total_cost(const std::vector<std::pair<Vector, Vector>>& data, double lmbda) {
    return this->total_cost(upload_dataset(data), lmbda);
}

double Network2::total_cost(const std::vector<std::pair<Vector, int>>& data, double lmbda) {
    return this->total_cost(upload_dataset(data), lmbda);
}

double Network2::total_cost(const DeviceDataset& data, double lmbda) {
    std::vector<float> costs = this->sample_costs(data, cost == Cost::CROSS_ENTROPY);

    double c = 0.0;
    for (int i = 0; i < costs.size(); i++) {
        double sample = costs[i];
        if (std::isnan(sample) || std::isinf(sample)) {
            sample = std::numeric_limits<double>::max();
        }
        c += sample / data.n;
    }

    // L2 regularization term
    double sum_sq = 0.0;
    for (int i = 0; i < weights.size(); i++) {
        sum_sq += sum_squares(weights[i].data(), weights[i].size());
    }
    c += 0.5 * (lmbda / data.n) * sum_sq;
    return c;
}

Matrix Network2::average_gradient(const std::vector<std::pair<Vector, Vector>>& data) {
    // average of dC/db over data for every layer, used in Chapter 5
    // all of data is one batch, so nabla_b is already summed over it
    this->upload_batch(data);
    this->backpropagation(batch_x.data(), batch_y.data(), data.size(), cost == Cost::CROSS_ENTROPY);

    Matrix gradient;
    for (int i = 0; i < nabla_b.size(); i++) {
        std::vector<float> sum = nabla_b[i].download();
        Vector layer(sum.size());
        for (int j = 0; j < sum.size(); j++) {
            layer[j] = sum[j] / data.size();
        }
        gradient.push_back(layer);
    }
    return gradient;
}

void Network2::save(const std::string& filename) {
    // plain text: sizes, cost, then biases and weights layer by layer
    std::ofstream ofs(filename);
    assert(ofs.is_open() && "Cannot open file to save network");

    Matrix b = this->get_biases();
    Tensor w = this->get_weights();

    ofs << std::setprecision(17);
    ofs << sizes.size();
    for (int i = 0; i < sizes.size(); i++) {
        ofs << " " << sizes[i];
    }
    ofs << "\n" << (cost == Cost::QUADRATIC ? "QuadraticCost" : "CrossEntropyCost") << "\n";

    for (int i = 0; i < b.size(); i++) {
        for (int j = 0; j < b[i].size(); j++) {
            ofs << b[i][j] << " ";
        }
        ofs << "\n";
    }

    for (int i = 0; i < w.size(); i++) {
        for (int j = 0; j < w[i].size(); j++) {
            for (int k = 0; k < w[i][j].size(); k++) {
                ofs << w[i][j][k] << " ";
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

    Matrix b(num_layers - 1);
    for (int i = 0; i < num_layers - 1; i++) {
        b[i].resize(sizes[i + 1]);
        for (int j = 0; j < sizes[i + 1]; j++) {
            ifs >> b[i][j];
        }
    }

    Tensor w(num_layers - 1);
    for (int i = 0; i < num_layers - 1; i++) {
        w[i].assign(sizes[i + 1], Vector(sizes[i]));
        for (int j = 0; j < sizes[i + 1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                ifs >> w[i][j][k];
            }
        }
    }
    ifs.close();

    net.set_parameters(b, w);
    return net;
}

}
