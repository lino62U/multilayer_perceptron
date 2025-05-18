#pragma once

#include "neuron.hpp"
#include <vector>
#include "Activations.hpp"

class Layer {
protected:
    vector<Neuron*> neurons;
    ActivationFunction activation;
    bool is_output_layer = false;
    bool softmax_enabled = false;

public:
    Layer(int num_neurons, ActivationFunction act, bool has_bias = true);
    ~Layer();
    
    void connectTo(Layer* next_layer);
    void computeOutputs();
    void computeDeltas(const vector<float>* targets = nullptr);
    void updateWeights();
    void applySoftmax();
    
    vector<float> getOutputs() const;
    void setInputs(const vector<float>& inputs);
    void setAsOutputLayer(bool softmax = false);
    
    size_t size() const { return neurons.size(); }
    Neuron* operator[](size_t index) { return neurons[index]; }
    const Neuron* operator[](size_t index) const { return neurons[index]; }

    // Nuevas funciones para guardar/cargar pesos
    void saveWeights(ofstream& file) const;
    void loadWeights(ifstream& file);
};