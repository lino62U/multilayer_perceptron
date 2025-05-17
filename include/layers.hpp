#pragma once

#include "neuron.hpp"
#include <vector>

class Layers {
private:
    std::vector<std::vector<Neuron*>> layers;

public:
    Layers();
    ~Layers();
    void buildNetwork(const std::vector<int>& structure, ActivationFunction act, float learningRate);

    std::vector<std::vector<Neuron*>>& getLayers();
    const std::vector<std::vector<Neuron*>>& getLayers() const;
};
