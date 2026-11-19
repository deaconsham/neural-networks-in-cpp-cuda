/*
Header file for the naive average darkness baseline from Chapter 1 (mnist_average_darkness.py).
Classifies a digit by which digit's average darkness in the training data is closest to it.
*/

#ifndef AVERAGE_DARKNESS_H
#define AVERAGE_DARKNESS_H

#include <iostream>
#include <vector>
#include <cmath>

#include "common.h"

Vector avg_darknesses(const std::vector<std::pair<Vector, int>>& training_data);

int guess_digit(const Vector& image, const Vector& avgs);

#endif
