#include "neuron.hpp"
#include "ActivationFunction.hpp"
#include <cmath>
#include <Activations.hpp>



Neuron::Neuron(bool bias, bool isOut, ActivationFunction act, float lr)
    : isBias(bias), isOutput(isOut), activation(act), learningRate(lr) {
    if (isBias) output = 1.0f;
}

void Neuron::forward() {
    if (isBias) {
        output = 1.0f;
        return;
    }
    float sum = 0.0f;
    // weigth and pointer to prev
    for (auto& set : prev) {
        sum += *(set.second) * set.first->output;
    }
    net = sum;

    if (activation.activar)
        output = activation.activar(sum);
    else
        output = sum;
}

void Neuron::computeOutputDelta(float target) {
    if (activation.derivar)
        delta = (output - target) * activation.derivar(output);
}

void Neuron::computeHiddenDelta() {
    float sum = 0.0f;
    for (auto& n : next) {
        sum += *(n.second) * n.first->delta;
    }
    if (activation.derivar)
        delta = activation.derivar(output) * sum;
}

void Neuron::updateWeights() {
    if (isBias) return;
    for (auto& input : prev) {
        *(input.second) -= learningRate * delta * input.first->output;
    }
}

void Neuron::setPrev(Neuron* from, float* weight) {
    prev.push_back({from, weight});
}

void Neuron::setNext(Neuron* to, float* weight) {
    next.push_back({to, weight});
}

void Neuron::setActivation(const ActivationFunction& act) {
    activation = act;
}

float Neuron::getOutput() const {
    return output;
}

float Neuron::getDelta() const {
    return delta;
}

float Neuron::getNet() const {
    return net;
}

bool Neuron::getIsBias() const {
    return isBias;
}

void Neuron::setIsBias(bool value) {
    isBias = value;
}

bool Neuron::getIsOutput() const {
    return isOutput;
}

void Neuron::setIsOutput(bool value) {
    isOutput = value;
}



// Getters
const std::vector<std::pair<Neuron*, float*>>& Neuron::getNext() const {
    return next;
}

const std::vector<std::pair<Neuron*, float*>>& Neuron::getPrev() const {
    return prev;
}

ActivationFunction Neuron::getActivation() const {
    return activation;
}

float Neuron::getLearningRate() const {
    return learningRate;
}

// Setters
void Neuron::setOutput(float val) {
    output = val;
}

void Neuron::setNet(float val) {
    net = val;
}

void Neuron::setDelta(float val) {
    delta = val;
}


void Neuron::setLearningRate(float rate) {
    learningRate = rate;
}
