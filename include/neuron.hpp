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
#include <Activations.hpp>

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;
namespace fs = std::filesystem;

struct Neuron {
    float output = 0.0f;
    float net_input = 0.0f;
    float delta = 0.0f;
    bool is_bias = false;
    vector<pair<Neuron*, float*>> inputs;
    vector<pair<Neuron*, float*>> outputs;
    ActivationFunction activation;

    Neuron(ActivationFunction act = Activations::Sigmoid());
    void computeOutput();
    void computeDelta(bool is_output_neuron, float target = 0.0f);
    void updateWeights();
};