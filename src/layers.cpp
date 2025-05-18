#include "layers.hpp"
#include <cstdlib>

#define RANDOM_SEED 42

// -------------------- Capa --------------------
Layer::Layer(int num_neurons, ActivationFunction act, bool has_bias) : activation(act) {
    for (int i = 0; i < num_neurons; ++i) {
        neurons.push_back(new Neuron(act));
    }
    if (has_bias) {
        Neuron* bias = new Neuron(act);
        bias->is_bias = true;
        bias->output = 1.0f;
        neurons.push_back(bias);
    }
}

Layer::~Layer() {
    for (auto& neuron : neurons) {
        for (auto& [_, weight] : neuron->inputs) {
            delete weight;
        }
        delete neuron;
    }
}

void Layer::connectTo(Layer* next_layer) {
    srand(RANDOM_SEED);
    for (auto& neuron : next_layer->neurons) {
        if (neuron->is_bias) continue;
        for (auto& prev_neuron : neurons) {
            float* weight = new float((rand()%100)/100.0f - 0.5f);
            neuron->inputs.emplace_back(prev_neuron, weight);
            prev_neuron->outputs.emplace_back(neuron, weight);
        }
    }
}

void Layer::computeOutputs() {
    for (auto& neuron : neurons) {
        neuron->computeOutput();
    }
}

void Layer::computeDeltas(const vector<float>* targets) {
    if (is_output_layer) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            if (!neurons[i]->is_bias) {
                neurons[i]->computeDelta(true, targets ? (*targets)[i] : 0.0f);
            }
        }
    } else {
        for (auto& neuron : neurons) {
            if (!neuron->is_bias) {
                neuron->computeDelta(false);
            }
        }
    }
}

void Layer::updateWeights() {
    for (auto& neuron : neurons) {
        neuron->updateWeights();
    }
}

void Layer::applySoftmax() {
    if (!softmax_enabled) return;
    
    vector<float> z;
    for (auto& neuron : neurons) {
        if (!neuron->is_bias) {
            z.push_back(neuron->net_input);
        }
    }

    auto softmax_values = Activations::softmax(z);
    
    size_t idx = 0;
    for (auto& neuron : neurons) {
        if (!neuron->is_bias) {
            neuron->output = softmax_values[idx++];
        }
    }
}

vector<float> Layer::getOutputs() const {
    vector<float> output;
    for (auto& neuron : neurons) {
        if (!neuron->is_bias) {
            output.push_back(neuron->output);
        }
    }
    return output;
}

void Layer::setInputs(const vector<float>& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        neurons[i]->output = inputs[i];
    }
}

void Layer::setAsOutputLayer(bool softmax) {
    is_output_layer = true;
    softmax_enabled = softmax;
}

void Layer::saveWeights(ofstream& file) const {
    for (auto& neuron : neurons) {
        if (neuron->is_bias) continue;
        for (auto& [_, weight] : neuron->inputs) {
            file.write(reinterpret_cast<const char*>(weight), sizeof(float));
        }
    }
}

void Layer::loadWeights(ifstream& file) {
    for (auto& neuron : neurons) {
        if (neuron->is_bias) continue;
        for (auto& [_, weight] : neuron->inputs) {
            file.read(reinterpret_cast<char*>(weight), sizeof(float));
        }
    }
}
