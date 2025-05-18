#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;
namespace fs = std::filesystem;


class MNISTDataset {
public:
    static vector<vector<float>> loadImages(const string& filename, int max_images = -1);
    static vector<vector<float>> loadLabels(const string& filename, int max_labels = -1);
    static void displayImage(const vector<float>& image, int rows = 28, int cols = 28);
};
