/*
Header file for the types and helpers shared by every network.
*/

#ifndef COMMON_H
#define COMMON_H

#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Tensor = std::vector<Matrix>;
using Tensor4D = std::vector<Tensor>;
using IndexTensor = std::vector<std::vector<std::vector<int>>>;

// index of the highest value, the digit a network guessed
int argmax(const Vector& v);

// one-hot vector with a 1.0 at index j
Vector vectorized_result(int j);

#endif
