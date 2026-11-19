/*
Naive average darkness baseline from Chapter 1 (mnist_average_darkness.py) on GPU (CUDA).
Classifies a digit by which digit's average darkness in the training data is closest to it.
*/

#include "average_darkness.cuh"

namespace gpu {

// one block per image, darkness of an image is the sum of its pixel intensities
__global__ void image_darkness_kernel(const float* images, int dim, float* darkness) {
    const float* image = images + blockIdx.x * dim;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        sum += image[j];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        darkness[blockIdx.x] = sum;
    }
}

// one block per digit, sums the darkness of every image with that label
__global__ void digit_darkness_kernel(const float* darkness, const int* labels, int n, double* avgs) {
    int d = blockIdx.x;
    double sum = 0.0;
    double count = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (labels[i] == d) {
            sum += darkness[i];
            count += 1.0;
        }
    }
    sum = block_reduce_sum(sum);
    count = block_reduce_sum(count);
    if (threadIdx.x == 0) {
        avgs[d] = sum / count;
    }
}

// one thread per image, the digit whose average darkness is closest
__global__ void guess_kernel(const float* darkness, const double* avgs, int n, int* guesses) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    double dark = darkness[i];
    int best = 0;
    for (int d = 1; d < 10; d++) {
        if (fabs(avgs[d] - dark) < fabs(avgs[best] - dark)) {
            best = d;
        }
    }
    guesses[i] = best;
}

static DeviceBuffer<float> image_darknesses(const DeviceDataset& data) {
    DeviceBuffer<float> darkness(data.n);
    image_darkness_kernel<<<data.n, BLOCK_SIZE>>>(data.images.data(), data.dim, darkness.data());
    CUDA_CHECK_LAUNCH();
    return darkness;
}

Vector avg_darknesses(const std::vector<std::pair<Vector, int>>& training_data) {
    DeviceDataset data = upload_dataset(training_data);
    DeviceBuffer<float> darkness = image_darknesses(data);

    DeviceBuffer<double> avgs(10);
    digit_darkness_kernel<<<10, BLOCK_SIZE>>>(darkness.data(), data.labels.data(), data.n, avgs.data());
    CUDA_CHECK_LAUNCH();

    std::vector<double> host = avgs.download();
    return Vector(host.begin(), host.end());
}

std::vector<int> guess_digits(const std::vector<std::pair<Vector, int>>& data, const Vector& avgs) {
    DeviceDataset d = upload_dataset(data);
    DeviceBuffer<float> darkness = image_darknesses(d);
    DeviceBuffer<double> avgs_dev(avgs);

    DeviceBuffer<int> guesses(d.n);
    guess_kernel<<<num_blocks(d.n), BLOCK_SIZE>>>(darkness.data(), avgs_dev.data(), d.n, guesses.data());
    CUDA_CHECK_LAUNCH();
    return guesses.download();
}

}
