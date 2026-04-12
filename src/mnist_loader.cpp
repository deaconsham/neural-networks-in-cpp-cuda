/*
To load the MNIST file and covert pixels into float values.
*/

#include "mnist_loader.h"

MNIST_Loader::MNIST_Loader(std::string image_file, std::string label_file, int num) {
    load_images(image_file, num);
    load_labels(label_file, num);
}

MNIST_Loader::MNIST_Loader(std::string image_file, std::string label_file) {
    load_images(image_file, 0);
    load_labels(label_file, 0);
}

void MNIST_Loader::load_images(std::string file, int num) {
    std::ifstream ifs(file, std::ios::binary);
    assert(ifs.is_open() && "Cannot open image file.");

    char p[4];
    ifs.read(p, 4);

    assert(to_int(p) == 2051 && "Invalid image magic number");

    ifs.read(p, 4);
    int total_images = to_int(p);

    ifs.read(p, 4);
    int rows = to_int(p);
    
    ifs.read(p, 4);
    int cols = to_int(p);

    if (num > 0 && num <= total_images) {
        size = num;
    } else {
        size = total_images;
    }

    images.resize(size, std::vector<double>(rows * cols));

    int image_size = rows * cols;
    std::vector<unsigned char> buffer(image_size);

    for (int i = 0; i < size; i++) {
        ifs.read(reinterpret_cast<char*>(buffer.data()), image_size);
        for (int j = 0; j < image_size; j++) {
            images[i][j] = buffer[j] / 255.0;
        }
    }
    ifs.close();
}

void MNIST_Loader::load_labels(std::string file, int num) {
    std::ifstream ifs(file, std::ios::binary);

    assert(ifs.is_open() && "Cannot open label file");

    char p[4];
    ifs.read(p, 4);

    assert(to_int(p) == 2049 && "Invalid label magic number");

    ifs.read(p, 4);
    int total_labels = to_int(p);

    int label_size;
    if (num > 0 && num <= total_labels) {
        label_size = num;
    } else {
        label_size = total_labels;
    }

    assert(label_size == size && "Mismatch between number of images and labels");

    labels.resize(size);

    std::vector<unsigned char> buffer(size);
    ifs.read(reinterpret_cast<char*>(buffer.data()), size);

    for (int i = 0; i < size; i++) {
        labels[i] = static_cast<int>(buffer[i]);
    }
    ifs.close();
}

int MNIST_Loader::to_int(char* p) {
    return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) | ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}