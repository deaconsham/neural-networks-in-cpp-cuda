# neural-networks-in-cpp-cuda

MNIST neural network training C++ and Standard Library implementation, and an Nvidia CUDA implementation that trains on the GPU.
Based on [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen.
Original implementation was using Python and NumPy without PyTorch, and Theano for the convolutional network.
Every network here is written from scratch, including backpropagation through the convolutional, pooling and dropout layers.
The CUDA version doesn't use cuBLAS or cuDNN either, every kernel including matrix multiplication is hand-written.

## Chapters

| Mode | Chapter | Original file | C++ files | CUDA files | What it does |
|------|---------|---------------|-----------|------------|--------------|
| `darkness` | 1 | `mnist_average_darkness.py` | `average_darkness.h/.cpp` | `average_darkness.cuh/.cu` | Naive baseline, guesses a digit by how dark the image is |
| `ch1` | 1 & 2 | `network.py` | `network.h/.cpp` | `network.cuh/.cu` | Fully-connected sigmoid network, quadratic cost, SGD and backpropagation |
| `ch3` | 3 | `network2.py` | `network2.h/.cpp` | `network2.cuh/.cu` | Cross-entropy cost, L2 regularization, better weight initialization, monitoring, early stopping, save / load |
| `ch5` | 5 | `generate_gradient.py` | `network2.h/.cpp` | `network2.cuh/.cu` | Vanishing gradient problem, measures how fast each hidden layer learns |
| `ch6` | 6 | `network3.py`, `expand_mnist.py` | `conv_network.h/.cpp`, `expand_mnist.h/.cpp` | `conv_network.cuh/.cu` | Final convolutional network with ReLU, dropout and expanded training data |

### Chapter 1: average darkness baseline

Averages the total pixel darkness of each digit over the training data, then guesses whichever digit's average is closest.
Gets 2,225 / 10,000 correct, which is the number any real network has to beat.

### Chapter 1 & 2: network

`784 -> 30 -> 10` sigmoid network trained with the quadratic cost, η = 3.0, mini-batches of 10 for 30 epochs.
Reaches about 95% test accuracy.

### Chapter 3: network2

`784 -> 100 -> 10` sigmoid network trained with:

- **Cross-entropy cost**: the output error becomes `a - y`, so the `σ'(z)` term that slows learning disappears
- **L2 regularization**: weights are rescaled by `1 - ηλ/n` each step to reduce overfitting
- **Weight initialization** from `N(0, 1/n_in)` so hidden neurons don't start saturated
- **Monitoring** of cost and accuracy on training and evaluation data each epoch, and optional **early stopping**
- **Save / load** of the trained network to `network2.txt`

Uses η = 0.5, λ = 5.0, mini-batches of 10 for 30 epochs, validated on the 10,000 held-out training images.
Reaches about 98% accuracy.

### Chapter 5: vanishing gradient

Builds networks with 2, 3 and 4 hidden layers of 30 neurons and prints the length of `∂C/∂b` for each hidden layer while training on 1,000 images.
Each earlier layer learns about 5x slower than the one after it, which is why deep sigmoid networks are hard to train:

```
Network with 4 hidden layers of 30 neurons
Epoch 0:  layer 1 = 0.00164844  layer 2 = 0.00960999  layer 3 = 0.0495265  layer 4 = 0.221984
```

### Chapter 6: convolutional network

The final network from the textbook, which gets about 99.6% test accuracy:

```
input            1 x 28 x 28
conv 5x5 + pool  20 x 24 x 24  ->  20 x 12 x 12   (ReLU)
conv 5x5 + pool  40 x 8 x 8    ->  40 x 4 x 4     (ReLU)
fully-connected  640  -> 1000                     (ReLU, dropout 0.5)
fully-connected  1000 -> 1000                     (ReLU, dropout 0.5)
softmax          1000 -> 10                       (dropout 0.5)
```

- **Convolutional-pooling layers**: shared 5x5 filters followed by 2x2 max-pooling. Backpropagation only sends gradient through the max of each pooling window
- **ReLU** activations instead of sigmoid
- **Dropout** on the fully-connected and softmax layers during training. At test time the weights are scaled by `1 - p` instead
- **Log-likelihood cost** with a softmax output layer
- **L2 regularization** with λ = 0.1
- **Expanded training data**: each of the 50,000 training images is also shifted one pixel up, down, left and right, giving 250,000 images

Trained with η = 0.03, mini-batches of 10 for 40 epochs. Each epoch reports validation accuracy, and the test accuracy is checked whenever validation accuracy improves.

On CPU this takes about 10 minutes per epoch (around 7 hours total), so lower `epochs` in `src/main.cpp` for a quicker run, or use the CUDA version.

## Project structure

```
inc/
  common.h              Vector / Matrix / Tensor types and helpers shared by every network
  mnist_loader.h        reads the MNIST idx files
  average_darkness.h    Chapter 1 baseline
  network.h             Chapter 1 & 2 network
  network2.h            Chapter 3 network, also used for Chapter 5
  conv_network.h        Chapter 6 ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer and ConvNetwork
  expand_mnist.h        expands the training data for Chapter 6
src/
  *.cpp                 implementations of the headers above
  main.cpp              runs a chapter
cuda/
  inc/
    common.cuh          GPU memory buffer, datasets on the GPU, matrix multiplication and reduction kernels
    average_darkness.cuh
    network.cuh         also has SigmoidMLP, the layers shared by network and network2
    network2.cuh
    conv_network.cuh
  src/
    *.cu                implementations of the headers above
    main.cu             runs a chapter on the GPU
mnist/                  MNIST training and test data
```

The 60,000 MNIST training images are split into 50,000 training and 10,000 validation images, matching the `mnist.pkl.gz` file used in the textbook. Chapter 1 trains on all 60,000.

## Build and run

### CPU

```bash
g++ -std=c++17 -O3 -Iinc src/*.cpp -o neural_network

./neural_network darkness   # Chapter 1 baseline
./neural_network ch1        # Chapter 1 & 2
./neural_network ch3        # Chapter 3
./neural_network ch5        # Chapter 5
./neural_network ch6        # Chapter 6 (default)
```

Works with both GCC and Clang. `-O3` matters a lot for training speed.

### CUDA

Needs an Nvidia GPU and the CUDA Toolkit 11.6 or later. With an older toolkit, replace `-arch=native` with your GPU's architecture, e.g. `-arch=sm_75`.

```bash
nvcc -std=c++17 -O3 -arch=native -Iinc -Icuda/inc \
    src/common.cpp src/mnist_loader.cpp src/expand_mnist.cpp cuda/src/*.cu \
    -o neural_network_cuda

./neural_network_cuda darkness   # Chapter 1 baseline
./neural_network_cuda ch1        # Chapter 1 & 2
./neural_network_cuda ch3        # Chapter 3
./neural_network_cuda ch5        # Chapter 5
./neural_network_cuda ch6        # Chapter 6 (default)
```

The MNIST loader and `expand_mnist` are shared with the CPU version.

## CUDA implementation

Same networks, hyperparameters and output as the CPU version, with these differences:

- **Mini-batches as matrices**: instead of backpropagating one image at a time and summing, a mini-batch is one matrix with a row per image, so `nabla_w = δᵀ · a` sums the gradient over the batch in a single matrix multiplication
- **Tiled matrix multiplication**: each block loads 16x16 tiles into shared memory, so every value read from GPU memory is used 16 times
- **Data stays on the GPU**: the training data is uploaded once, and each epoch shuffles a list of indices that a kernel uses to gather the next mini-batch
- **float instead of double**: most GPUs are much slower with double. The cross-entropy cost is calculated from `z` with `softplus` so it never takes `log(0)` in float
- **Convolution backpropagation**: one thread per weight sums its own gradient, and the gradient to the previous layer uses `atomicAdd` since pooling windows overlap on the input
- **Dropout masks** come from a hash of a seed and the index, so every thread gets its own random number without storing generator state

`network2.txt` uses the same format on both, so a network saved by one can be loaded by the other.
Chapter 6 keeps the 250,000 expanded training images on the GPU, which needs about 1 GB of GPU memory.
