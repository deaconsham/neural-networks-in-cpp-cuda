/*
Header file for the naive average darkness baseline from Chapter 1 (mnist_average_darkness.py) on GPU (CUDA).
Classifies a digit by which digit's average darkness in the training data is closest to it.
*/

#ifndef CUDA_AVERAGE_DARKNESS_CUH
#define CUDA_AVERAGE_DARKNESS_CUH

#include <iostream>
#include <vector>
#include <cmath>

#include "common.h"
#include "common.cuh"

namespace gpu {

Vector avg_darknesses(const std::vector<std::pair<Vector, int>>& training_data);

// guesses every image at once
std::vector<int> guess_digits(const std::vector<std::pair<Vector, int>>& data, const Vector& avgs);

}

#endif
