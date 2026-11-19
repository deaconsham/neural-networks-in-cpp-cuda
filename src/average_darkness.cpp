/*
Naive average darkness baseline from Chapter 1 (mnist_average_darkness.py).
Classifies a digit by which digit's average darkness in the training data is closest to it.
*/

#include "average_darkness.h"

Vector avg_darknesses(const std::vector<std::pair<Vector, int>>& training_data) {
    Vector darkness(10, 0.0);
    Vector count(10, 0.0);

    // darkness of an image is the sum of its pixel intensities
    for (int i = 0; i < training_data.size(); i++) {
        double sum = 0.0;
        for (int j = 0; j < training_data[i].first.size(); j++) {
            sum += training_data[i].first[j];
        }
        darkness[training_data[i].second] += sum;
        count[training_data[i].second] += 1.0;
    }

    for (int d = 0; d < 10; d++) {
        darkness[d] /= count[d];
    }
    return darkness;
}

int guess_digit(const Vector& image, const Vector& avgs) {
    double darkness = 0.0;
    for (int i = 0; i < image.size(); i++) {
        darkness += image[i];
    }

    int best = 0;
    for (int d = 1; d < 10; d++) {
        if (std::fabs(avgs[d] - darkness) < std::fabs(avgs[best] - darkness)) {
            best = d;
        }
    }
    return best;
}
