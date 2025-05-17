#include "layers.hpp"
#include <cstdlib>

#define RANDOM_SEED 42

Layers::Layers() {}

Layers::~Layers() {
    for (auto& layer : layers) {
        for (auto& neuron : layer) {
            delete neuron;
        }
    }
}

void Layers::buildNetwork(const std::vector<int>& structure, ActivationFunction act, float learningRate){
    srand(RANDOM_SEED);

    for (int i = 0; i < structure.size(); ++i) {
        std::vector<Neuron*> layer;
        int numNeurons = structure[i];
        if (i != structure.size() - 1) numNeurons += 1;

        for (int j = 0; j < numNeurons; ++j) {
            Neuron* n = new Neuron(false, false, act, learningRate);  // Usa el constructor con valores explícitos
            
            if (i != structure.size() - 1 && j == numNeurons - 1) {
                n->setIsBias(true);
                n->setIsOutput( 1.0f);
            }
            layer.push_back(n);
        }
        layers.push_back(layer);
    }

    for (int i = 1; i < layers.size(); ++i) {
        for (auto& neuron : layers[i]) {
            if (neuron->getIsBias()) continue;
            for (auto& previous : layers[i - 1]) {
               // float* weight = new float(0.0f);
                float* weight = new float(((rand() % 100) / 100.0f) - 0.5f); // Peso entre -0.5 y 0.5
                neuron->setPrev(previous, weight);
                previous->setNext(neuron, weight);
            }
        }
    }
}

std::vector<std::vector<Neuron*>>& Layers::getLayers() {
    return layers;
}

const std::vector<std::vector<Neuron*>>& Layers::getLayers() const {
    return layers;
}
