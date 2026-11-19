/*
Types and helpers shared by every network.
*/

#include "common.h"

int argmax(const Vector& v) {
    int max_idx = 0;
    for (int i = 1; i < v.size(); i++) {
        if (v[i] > v[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

Vector vectorized_result(int j) {
    Vector arr(10, 0.0);
    arr[j] = 1.0;
    return arr;
}
