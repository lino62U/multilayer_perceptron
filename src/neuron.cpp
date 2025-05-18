#include "neuron.hpp"
#include "Activations.hpp"
#include <cmath>


// -------------------- Neurona --------------------
Neuron::Neuron(ActivationFunction act) : activation(act) {}

void Neuron::computeOutput() {
    if (is_bias) {
        output = 1.0f;
        return;
    }
    
    net_input = 0.0f;
    for (const auto& [neuron, weight] : inputs) {
        net_input += *weight * neuron->output;
    }
    
    if (activation.activate) {
        output = activation.activate(net_input);
    }
}

void Neuron::computeDelta(bool is_output_neuron, float target) {
    if (is_output_neuron) {
        delta = (output - target); // Para softmax, la derivada ya está considerada
    } else {
        float sum = 0.0f;
        for (const auto& [neuron, weight] : outputs) {
            sum += *weight * neuron->delta;
        }
        delta = activation.derive(output) * sum;
    }
}

void Neuron::updateWeights() {
    if (is_bias) return;
    for (auto& [neuron, weight] : inputs) {
        *weight -= LEARNING_RATE * delta * neuron->output;
    }
}

