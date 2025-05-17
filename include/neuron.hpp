#pragma once
#include <vector>
#include <utility>
#include "ActivationFunction.hpp"

class Neuron {
private:
    float output = 0.0f, net = 0.0f, delta = 0.0f;
    bool isBias = false, isOutput = false;

    std::vector<std::pair<Neuron*, float*>> next, prev;
    ActivationFunction activation;
    float learningRate = 0.1f;  // Ahora es miembro no estático

public:
  
    Neuron(bool bias = false, bool isOutput = false,
           ActivationFunction act = ActivationFunction(nullptr, nullptr),
           float lr = 0.1f);

    void forward();
    void computeOutputDelta(float target);
    void computeHiddenDelta();
    void updateWeights();

    void setPrev(Neuron* from, float* weight);
    void setNext(Neuron* to, float* weight);

    // Getters
    float getOutput() const;
    float getNet() const;
    float getDelta() const;
    bool getIsBias() const;
    bool getIsOutput() const;
    const std::vector<std::pair<Neuron*, float*>>& getNext() const;
    const std::vector<std::pair<Neuron*, float*>>& getPrev() const;
    ActivationFunction getActivation() const;
    float getLearningRate() const;

    // Setters
    void setOutput(float val);
    void setNet(float val);
    void setDelta(float val);
    void setIsBias(bool val);
    void setIsOutput(bool val);
   
    void setActivation(const ActivationFunction& act);
    void setLearningRate(float rate);

};
