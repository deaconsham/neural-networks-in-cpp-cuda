/*
To expand the MNIST training data from 50,000 to 250,000 images.
Each image is kept and also shifted by one pixel up, down, left and right.
*/

#include "expand_mnist.h"

std::vector<std::pair<Vector, int>> expand_data(const std::vector<std::pair<Vector, int>>& data, int rows, int cols) {
    std::vector<std::pair<Vector, int>> expanded;
    expanded.reserve(data.size() * 5);

    // {row shift, col shift}: down, up, right, left
    int shifts[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int i = 0; i < data.size(); i++) {
        expanded.push_back(data[i]);
        for (int s = 0; s < 4; s++) {
            Vector shifted = shift_image(data[i].first, rows, cols, shifts[s][0], shifts[s][1]);
            expanded.push_back(std::make_pair(shifted, data[i].second));
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(expanded.begin(), expanded.end(), g);

    return expanded;
}

Vector shift_image(const Vector& image, int rows, int cols, int d_row, int d_col) {
    // pixels shifted in from outside the image are left as zero
    Vector arr(rows * cols, 0.0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int src_i = i - d_row;
            int src_j = j - d_col;
            if (src_i >= 0 && src_i < rows && src_j >= 0 && src_j < cols) {
                arr[i * cols + j] = image[src_i * cols + src_j];
            }
        }
    }
    return arr;
}
