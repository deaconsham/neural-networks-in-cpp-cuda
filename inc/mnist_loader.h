/*
Header file to load the MNIST file and covert pixels into float values.
*/

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Tensor = std::vector<Matrix>;

class MNIST_Loader {
public:
  std::vector<std::vector<double>> images;
  std::vector<int> labels;
  int size;
  int rows;
  int cols;

  MNIST_Loader(std::string image_file, std::string label_file, int num);

  MNIST_Loader(std::string image_file, std::string label_file);

  void load_images(std::string file, int num=0);

  void load_labels(std::string file, int num=0);

  int to_int(char* p);
};

#endif