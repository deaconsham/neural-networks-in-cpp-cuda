/*
Final convolutional neural network from Chapter 6 (network3.py) on CPU.
Architecture: convolutional-pooling layers, fully-connected layers and a softmax output layer.
Uses ReLU activations, dropout, L2 regularization and the log-likelihood cost.
Gradients are calculated using backpropagation.
*/

#include "conv_network.h"

// convolutional + max-pooling layer

ConvPoolLayer::ConvPoolLayer(const std::vector<int>& image_shape, const std::vector<int>& filter_shape, int pool_size, std::mt19937& gen) {
    assert(image_shape[0] == filter_shape[1] && "Image channels must match filter channels");

    num_filters = filter_shape[0];
    in_channels = filter_shape[1];
    filter_rows = filter_shape[2];
    filter_cols = filter_shape[3];

    in_rows = image_shape[1];
    in_cols = image_shape[2];

    this->pool_size = pool_size;

    // valid convolution shrinks the image, pooling divides it
    conv_rows = in_rows - filter_rows + 1;
    conv_cols = in_cols - filter_cols + 1;
    out_rows = conv_rows / pool_size;
    out_cols = conv_cols / pool_size;

    // weights ~ N(0, 1/n_out), biases ~ N(0, 1)
    double n_out = static_cast<double>(num_filters * filter_rows * filter_cols) / (pool_size * pool_size);
    std::normal_distribution<double> w_dis(0.0, std::sqrt(1.0 / n_out));
    std::normal_distribution<double> b_dis(0.0, 1.0);

    weights.resize(num_filters, Tensor(in_channels, Matrix(filter_rows, Vector(filter_cols))));
    for (int f = 0; f < num_filters; f++) {
        for (int ch = 0; ch < in_channels; ch++) {
            for (int i = 0; i < filter_rows; i++) {
                for (int j = 0; j < filter_cols; j++) {
                    weights[f][ch][i][j] = w_dis(gen);
                }
            }
        }
    }

    biases.resize(num_filters);
    for (int f = 0; f < num_filters; f++) {
        biases[f] = b_dis(gen);
    }

    nabla_w = weights;
    nabla_b = biases;
    this->zero_gradients();
}

Tensor ConvPoolLayer::feed_forward(const Tensor& x) {
    input = x;
    z.assign(num_filters, Matrix(out_rows, Vector(out_cols)));
    max_idx.assign(num_filters, std::vector<std::vector<int>>(out_rows, std::vector<int>(out_cols)));
    Tensor output(num_filters, Matrix(out_rows, Vector(out_cols)));

    Matrix conv(conv_rows, Vector(conv_cols));

    for (int f = 0; f < num_filters; f++) {

        // convolution

        for (int i = 0; i < conv_rows; i++) {
            std::fill(conv[i].begin(), conv[i].end(), 0.0);
        }

        for (int ch = 0; ch < in_channels; ch++) {
            for (int kr = 0; kr < filter_rows; kr++) {
                for (int kc = 0; kc < filter_cols; kc++) {
                    double w = weights[f][ch][kr][kc];
                    for (int i = 0; i < conv_rows; i++) {
                        for (int j = 0; j < conv_cols; j++) {
                            conv[i][j] += w * x[ch][i + kr][j + kc];
                        }
                    }
                }
            }
        }

        // max pooling, then bias and activation

        for (int pi = 0; pi < out_rows; pi++) {
            for (int pj = 0; pj < out_cols; pj++) {
                double max_val = -std::numeric_limits<double>::infinity();
                int best = 0;
                for (int di = 0; di < pool_size; di++) {
                    for (int dj = 0; dj < pool_size; dj++) {
                        int r = pi * pool_size + di;
                        int c = pj * pool_size + dj;
                        if (conv[r][c] > max_val) {
                            max_val = conv[r][c];
                            best = r * conv_cols + c;
                        }
                    }
                }
                max_idx[f][pi][pj] = best;
                z[f][pi][pj] = max_val + biases[f];
                output[f][pi][pj] = relu(z[f][pi][pj]);
            }
        }
    }
    return output;
}

Tensor ConvPoolLayer::backpropagation(const Tensor& delta) {
    Tensor delta_input(in_channels, Matrix(in_rows, Vector(in_cols, 0.0)));

    for (int f = 0; f < num_filters; f++) {
        for (int pi = 0; pi < out_rows; pi++) {
            for (int pj = 0; pj < out_cols; pj++) {
                double dz = delta[f][pi][pj] * relu_prime(z[f][pi][pj]);
                if (dz == 0.0) {
                    continue;
                }

                nabla_b[f] += dz;

                // only the max of each pooling window received input, so only it gets gradient
                int r = max_idx[f][pi][pj] / conv_cols;
                int c = max_idx[f][pi][pj] % conv_cols;

                for (int ch = 0; ch < in_channels; ch++) {
                    for (int kr = 0; kr < filter_rows; kr++) {
                        for (int kc = 0; kc < filter_cols; kc++) {
                            nabla_w[f][ch][kr][kc] += dz * input[ch][r + kr][c + kc];
                            delta_input[ch][r + kr][c + kc] += dz * weights[f][ch][kr][kc];
                        }
                    }
                }
            }
        }
    }
    return delta_input;
}

void ConvPoolLayer::zero_gradients() {
    for (int f = 0; f < num_filters; f++) {
        nabla_b[f] = 0.0;
        for (int ch = 0; ch < in_channels; ch++) {
            for (int i = 0; i < filter_rows; i++) {
                for (int j = 0; j < filter_cols; j++) {
                    nabla_w[f][ch][i][j] = 0.0;
                }
            }
        }
    }
}

void ConvPoolLayer::update(double learning_rate, int mini_batch_size, double weight_decay) {
    double factor = learning_rate / mini_batch_size;
    double decay = 1.0 - learning_rate * weight_decay;

    for (int f = 0; f < num_filters; f++) {
        biases[f] -= factor * nabla_b[f];
        for (int ch = 0; ch < in_channels; ch++) {
            for (int i = 0; i < filter_rows; i++) {
                for (int j = 0; j < filter_cols; j++) {
                    weights[f][ch][i][j] = decay * weights[f][ch][i][j] - factor * nabla_w[f][ch][i][j];
                }
            }
        }
    }
}

// fully-connected layer

FullyConnectedLayer::FullyConnectedLayer(int n_in, int n_out, double p_dropout) {
    this->n_in = n_in;
    this->n_out = n_out;
    this->p_dropout = p_dropout;

    weights.resize(n_out, Vector(n_in, 0.0));
    biases.resize(n_out, 0.0);

    nabla_w.resize(n_out, Vector(n_in, 0.0));
    nabla_b.resize(n_out, 0.0);
}

FullyConnectedLayer::FullyConnectedLayer(int n_in, int n_out, std::mt19937& gen, double p_dropout)
    : FullyConnectedLayer(n_in, n_out, p_dropout)
{
    // weights ~ N(0, 1/n_out), biases ~ N(0, 1)
    std::normal_distribution<double> w_dis(0.0, std::sqrt(1.0 / n_out));
    std::normal_distribution<double> b_dis(0.0, 1.0);

    for (int j = 0; j < n_out; j++) {
        biases[j] = b_dis(gen);
        for (int k = 0; k < n_in; k++) {
            weights[j][k] = w_dis(gen);
        }
    }
}

Vector FullyConnectedLayer::weighted_input(const Vector& x, double scale) {
    Vector arr(n_out);
    for (int j = 0; j < n_out; j++) {
        double sum = 0.0;
        for (int k = 0; k < n_in; k++) {
            sum += weights[j][k] * x[k];
        }
        arr[j] = scale * sum + biases[j];
    }
    return arr;
}

Vector FullyConnectedLayer::weighted_input_dropout(const Vector& x, std::mt19937& gen) {
    std::bernoulli_distribution keep(1.0 - p_dropout);

    // randomly drop inputs to this layer
    mask.resize(n_in);
    input.resize(n_in);
    for (int k = 0; k < n_in; k++) {
        mask[k] = keep(gen) ? 1.0 : 0.0;
        input[k] = x[k] * mask[k];
    }

    z = this->weighted_input(input, 1.0);
    return z;
}

Vector FullyConnectedLayer::backpropagate_dz(const Vector& dz) {
    Vector delta_input(n_in, 0.0);

    for (int j = 0; j < n_out; j++) {
        if (dz[j] == 0.0) {
            continue;
        }
        nabla_b[j] += dz[j];
        for (int k = 0; k < n_in; k++) {
            nabla_w[j][k] += dz[j] * input[k];
            delta_input[k] += weights[j][k] * dz[j];
        }
    }

    // dropped inputs did not affect the output
    for (int k = 0; k < n_in; k++) {
        delta_input[k] *= mask[k];
    }
    return delta_input;
}

Vector FullyConnectedLayer::feed_forward(const Vector& x) {
    // no dropout at inference, so scale weights by the keep probability instead
    Vector arr = this->weighted_input(x, 1.0 - p_dropout);
    for (int j = 0; j < n_out; j++) {
        arr[j] = relu(arr[j]);
    }
    return arr;
}

Vector FullyConnectedLayer::feed_forward_dropout(const Vector& x, std::mt19937& gen) {
    Vector arr = this->weighted_input_dropout(x, gen);
    for (int j = 0; j < n_out; j++) {
        arr[j] = relu(arr[j]);
    }
    return arr;
}

Vector FullyConnectedLayer::backpropagation(const Vector& delta) {
    Vector dz(n_out);
    for (int j = 0; j < n_out; j++) {
        dz[j] = delta[j] * relu_prime(z[j]);
    }
    return this->backpropagate_dz(dz);
}

void FullyConnectedLayer::zero_gradients() {
    for (int j = 0; j < n_out; j++) {
        nabla_b[j] = 0.0;
        std::fill(nabla_w[j].begin(), nabla_w[j].end(), 0.0);
    }
}

void FullyConnectedLayer::update(double learning_rate, int mini_batch_size, double weight_decay) {
    double factor = learning_rate / mini_batch_size;
    double decay = 1.0 - learning_rate * weight_decay;

    for (int j = 0; j < n_out; j++) {
        biases[j] -= factor * nabla_b[j];
        for (int k = 0; k < n_in; k++) {
            weights[j][k] = decay * weights[j][k] - factor * nabla_w[j][k];
        }
    }
}

// softmax output layer

SoftmaxLayer::SoftmaxLayer(int n_in, int n_out, double p_dropout)
    : FullyConnectedLayer(n_in, n_out, p_dropout)    // softmax weights and biases start at zero
{
}

Vector SoftmaxLayer::feed_forward(const Vector& x) {
    return softmax(this->weighted_input(x, 1.0 - p_dropout));
}

Vector SoftmaxLayer::feed_forward_dropout(const Vector& x, std::mt19937& gen) {
    output = softmax(this->weighted_input_dropout(x, gen));
    return output;
}

Vector SoftmaxLayer::backpropagation(int y) {
    // log-likelihood cost with softmax gives delta = a - y
    Vector dz = output;
    dz[y] -= 1.0;
    return this->backpropagate_dz(dz);
}

// network

ConvNetwork::ConvNetwork(const std::vector<ConvPoolLayer>& conv_layers, const std::vector<FullyConnectedLayer>& fc_layers, const SoftmaxLayer& softmax_layer)
    : conv_layers(conv_layers), fc_layers(fc_layers), softmax_layer(softmax_layer)
{
    std::random_device rd;
    gen.seed(rd());
}

Vector ConvNetwork::conv_feed_forward(const Vector& x) {
    if (conv_layers.empty()) {
        return x;
    }

    ConvPoolLayer& first = conv_layers[0];
    Tensor image = to_image(x, first.in_channels, first.in_rows, first.in_cols);
    for (int i = 0; i < conv_layers.size(); i++) {
        image = conv_layers[i].feed_forward(image);
    }
    return flatten(image);
}

Vector ConvNetwork::feed_forward(const Vector& x) {
    Vector arr = this->conv_feed_forward(x);
    for (int i = 0; i < fc_layers.size(); i++) {
        arr = fc_layers[i].feed_forward(arr);
    }
    return softmax_layer.feed_forward(arr);
}

void ConvNetwork::sgd(
    std::vector<std::pair<Vector, int>>& training_data,
    int epochs,
    int mini_batch_size,
    double learning_rate,
    const std::vector<std::pair<Vector, int>>& validation_data,
    const std::vector<std::pair<Vector, int>>& test_data,
    double lmbda)
{
    int n = training_data.size();
    int num_training_batches = n / mini_batch_size;

    double best_validation_accuracy = 0.0;
    int best_iteration = 0;
    double test_accuracy = 0.0;

    for (int i = 0; i < epochs; i++) {
        std::shuffle(training_data.begin(), training_data.end(), gen);

        for (int j = 0; j < num_training_batches; j++) {
            int iteration = num_training_batches * i + j;
            if (iteration % 1000 == 0) {
                std::cout << "Training mini-batch number " << iteration << std::endl;
            }

            std::vector<std::pair<Vector, int>> mini_batch(
                training_data.begin() + j * mini_batch_size,
                training_data.begin() + (j + 1) * mini_batch_size
            );
            this->update_mini_batch(mini_batch, learning_rate, lmbda, num_training_batches);
        }

        double validation_accuracy = this->accuracy(validation_data) * 100.0;
        std::cout << "Epoch " << i + 1 << ": validation accuracy " << validation_accuracy << "%" << std::endl;

        // only check the test set when validation improves
        if (validation_accuracy >= best_validation_accuracy) {
            std::cout << "This is the best validation accuracy to date." << std::endl;
            best_validation_accuracy = validation_accuracy;
            best_iteration = num_training_batches * (i + 1) - 1;
            test_accuracy = this->accuracy(test_data) * 100.0;
            std::cout << "The corresponding test accuracy is " << test_accuracy << "%" << std::endl;
        }
    }

    std::cout << "Finished training network." << std::endl;
    std::cout << "Best validation accuracy of " << best_validation_accuracy << "% obtained at iteration " << best_iteration << std::endl;
    std::cout << "Corresponding test accuracy of " << test_accuracy << "%" << std::endl;
}

void ConvNetwork::update_mini_batch(std::vector<std::pair<Vector, int>>& mini_batch, double learning_rate, double lmbda, int num_training_batches) {
    for (int i = 0; i < conv_layers.size(); i++) {
        conv_layers[i].zero_gradients();
    }
    for (int i = 0; i < fc_layers.size(); i++) {
        fc_layers[i].zero_gradients();
    }
    softmax_layer.zero_gradients();

    // each call adds its gradient into the layers' nabla_w and nabla_b
    for (int i = 0; i < mini_batch.size(); i++) {
        this->backpropagation(mini_batch[i].first, mini_batch[i].second);
    }

    // L2 term 0.5 * lmbda * sum(w^2) / num_training_batches, as in network3.py
    double weight_decay = lmbda / num_training_batches;
    int m = mini_batch.size();

    for (int i = 0; i < conv_layers.size(); i++) {
        conv_layers[i].update(learning_rate, m, weight_decay);
    }
    for (int i = 0; i < fc_layers.size(); i++) {
        fc_layers[i].update(learning_rate, m, weight_decay);
    }
    softmax_layer.update(learning_rate, m, weight_decay);
}

void ConvNetwork::backpropagation(const Vector& x, int y) {

    // forward pass with dropout

    Vector arr = this->conv_feed_forward(x);
    for (int i = 0; i < fc_layers.size(); i++) {
        arr = fc_layers[i].feed_forward_dropout(arr, gen);
    }
    softmax_layer.feed_forward_dropout(arr, gen);

    // backward pass

    Vector delta = softmax_layer.backpropagation(y);

    for (int i = fc_layers.size() - 1; i >= 0; i--) {
        delta = fc_layers[i].backpropagation(delta);
    }

    if (!conv_layers.empty()) {
        ConvPoolLayer& last = conv_layers[conv_layers.size() - 1];
        Tensor delta_image = to_image(delta, last.num_filters, last.out_rows, last.out_cols);
        for (int i = conv_layers.size() - 1; i >= 0; i--) {
            delta_image = conv_layers[i].backpropagation(delta_image);
        }
    }
}

double ConvNetwork::accuracy(const std::vector<std::pair<Vector, int>>& data) {
    int count = 0;

    for (int i = 0; i < data.size(); i++) {
        if (argmax(this->feed_forward(data[i].first)) == data[i].second) {
            count++;
        }
    }
    return static_cast<double>(count) / data.size();
}

// helpers fcns

double relu(double z) {
    return z > 0.0 ? z : 0.0;
}

double relu_prime(double z) {
    return z > 0.0 ? 1.0 : 0.0;
}

Vector softmax(const Vector& z) {
    // subtract the max for numerical stability
    double max_val = z[0];
    for (int i = 1; i < z.size(); i++) {
        max_val = std::max(max_val, z[i]);
    }

    Vector arr(z.size());
    double sum = 0.0;
    for (int i = 0; i < z.size(); i++) {
        arr[i] = std::exp(z[i] - max_val);
        sum += arr[i];
    }
    for (int i = 0; i < z.size(); i++) {
        arr[i] /= sum;
    }
    return arr;
}

Tensor to_image(const Vector& x, int channels, int rows, int cols) {
    assert(x.size() == channels * rows * cols && "Vector size does not match image shape");

    Tensor image(channels, Matrix(rows, Vector(cols)));
    int idx = 0;
    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                image[ch][i][j] = x[idx++];
            }
        }
    }
    return image;
}

Vector flatten(const Tensor& t) {
    Vector arr;
    for (int ch = 0; ch < t.size(); ch++) {
        for (int i = 0; i < t[ch].size(); i++) {
            arr.insert(arr.end(), t[ch][i].begin(), t[ch][i].end());
        }
    }
    return arr;
}
