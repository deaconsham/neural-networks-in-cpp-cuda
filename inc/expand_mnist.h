/*
Header file to expand the MNIST training data from 50,000 to 250,000 images.
Each image is kept and also shifted by one pixel up, down, left and right.
*/

#ifndef EXPAND_MNIST_H
#define EXPAND_MNIST_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "common.h"

std::vector<std::pair<Vector, int>> expand_data(const std::vector<std::pair<Vector, int>>& data, int rows, int cols);

Vector shift_image(const Vector& image, int rows, int cols, int d_row, int d_col);

#endif
