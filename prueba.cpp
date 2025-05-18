#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;

// ==================== DECLARACIONES ====================
using ActivationFunc = float(*)(float);

struct ActivationFunction {
    ActivationFunc activate;
    ActivationFunc derive;

    ActivationFunction(ActivationFunc a = nullptr, ActivationFunc d = nullptr)
        : activate(a), derive(d) {}
};

namespace Activations {
    // Funciones de activación
    static float sigmoid(float x);
    static float sigmoid_derivative(float y);
    static float relu(float x);
    static float relu_derivative(float x);
    static float tanh_act(float x);
    static float tanh_derivative(float y);
    static vector<float> softmax(const vector<float>& z);
    
    // Funciones de construcción
    static ActivationFunction Sigmoid();
    static ActivationFunction ReLU();
    static ActivationFunction Tanh();
    static ActivationFunction Softmax();
}

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
};

class MultilayerPerceptron {
    vector<Layer*> layers;
    bool softmax_output = false;

public:
    ~MultilayerPerceptron();
    void createNetwork(const vector<int>& architecture, 
                      const vector<ActivationFunction>& activations);
    void setInput(const vector<float>& inputs);
    vector<float> forwardPropagate();
    void backPropagate(const vector<float>& targets);
    void train(const vector<float>& input, const vector<float>& target);
    void printNetwork() const;
};

// ==================== IMPLEMENTACIONES ====================

// -------------------- Funciones de Activación --------------------
namespace Activations {
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static float sigmoid_derivative(float y) {
        return y * (1.0f - y);
    }

    static float relu(float x) {
        return x > 0 ? x : 0.0f;
    }

    static float relu_derivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    static float tanh_act(float x) {
        return std::tanh(x);
    }

    static float tanh_derivative(float y) {
        return 1.0f - y * y;
    }

    static vector<float> softmax(const vector<float>& z) {
        vector<float> res(z.size());
        float max_z = *max_element(z.begin(), z.end());
        float sum = 0.0f;

        for (size_t i = 0; i < z.size(); ++i) {
            res[i] = exp(z[i] - max_z);
            sum += res[i];
        }

        for (auto& val : res) {
            val /= sum;
        }

        return res;
    }

    static ActivationFunction Sigmoid() {
        return ActivationFunction(sigmoid, sigmoid_derivative);
    }

    static ActivationFunction ReLU() {
        return ActivationFunction(relu, relu_derivative);
    }

    static ActivationFunction Tanh() {
        return ActivationFunction(tanh_act, tanh_derivative);
    }

    static ActivationFunction Softmax() {
        return ActivationFunction(nullptr, nullptr); // Caso especial
    }
}

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

// -------------------- Perceptrón Multicapa --------------------
MultilayerPerceptron::~MultilayerPerceptron() {
    for (auto& layer : layers) {
        delete layer;
    }
}

void MultilayerPerceptron::createNetwork(const vector<int>& architecture, 
                  const vector<ActivationFunction>& activations) {
    if (activations.size() != architecture.size() - 1) {
        throw runtime_error("Number of activation functions must match hidden layers + output");
    }
    
    softmax_output = (activations.back().activate == nullptr);

    // Crear capas
    for (size_t i = 0; i < architecture.size(); ++i) {
        bool has_bias = (i != architecture.size()-1);
        ActivationFunction act = (i == 0) ? Activations::Sigmoid() : activations[i-1];
        
        Layer* layer = new Layer(architecture[i], act, has_bias);
        if (i == architecture.size()-1) {
            layer->setAsOutputLayer(softmax_output);
        }
        layers.push_back(layer);
    }

    // Conectar capas
    for (size_t i = 0; i < layers.size()-1; ++i) {
        layers[i]->connectTo(layers[i+1]);
    }
}

void MultilayerPerceptron::setInput(const vector<float>& inputs) {
    layers[0]->setInputs(inputs);
}

vector<float> MultilayerPerceptron::forwardPropagate() {
    for (size_t i = 1; i < layers.size(); ++i) {
        layers[i]->computeOutputs();
        if (i == layers.size()-1 && softmax_output) {
            layers[i]->applySoftmax();
        }
    }
    return layers.back()->getOutputs();
}

void MultilayerPerceptron::backPropagate(const vector<float>& targets) {
    // Capa de salida
    layers.back()->computeDeltas(&targets);

    // Capas ocultas
    for (int i = layers.size()-2; i >= 0; --i) {
        layers[i]->computeDeltas();
    }

    // Actualizar pesos
    for (size_t i = 1; i < layers.size(); ++i) {
        layers[i]->updateWeights();
    }
}

void MultilayerPerceptron::train(const vector<float>& input, const vector<float>& target) {
    setInput(input);
    forwardPropagate();
    backPropagate(target);
}

void MultilayerPerceptron::printNetwork() const {
    for (size_t i = 0; i < layers.size(); ++i) {
        cout << "Layer " << i << " (";
        if (i == 0) cout << "Input";
        else if (i == layers.size()-1) cout << (softmax_output ? "Softmax" : "Output");
        else cout << "Hidden";
        cout << "):\n";
        
        for (size_t j = 0; j < layers[i]->size(); ++j) {
            const Neuron* neuron = (*layers[i])[j];
            cout << "  Output: " << neuron->output << ", Delta: " << neuron->delta;
            if (neuron->is_bias) cout << " [bias]";
            cout << endl;
        }
    }
}

// ==================== MAIN ====================
int main() {
    MultilayerPerceptron mlp;
    
    // Ejemplo XOR con sigmoid (salida única)
    cout << "XOR with Sigmoid/Tanh (single output):\n";
    mlp.createNetwork({2, 2, 1}, {Activations::Sigmoid(), Activations::Sigmoid()});

    vector<vector<float>> training_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<vector<float>> training_targets = {{0}, {1}, {1}, {0}};

    // Entrenamiento
    for (int epoch = 0; epoch < 3000; ++epoch) {
        for (size_t i = 0; i < training_inputs.size(); ++i) {
            mlp.train(training_inputs[i], training_targets[i]);
        }
    }

    cout << "\nXOR Results:\n";
    for (const auto& input : training_inputs) {
        mlp.setInput(input);
        auto output = mlp.forwardPropagate();
        cout << input[0] << " XOR " << input[1] << " = " << output[0] << endl;
    }

    // Ejemplo multiclase con softmax
    cout << "\nMulticlass example with Softmax:\n";
    MultilayerPerceptron mlp_softmax;
    mlp_softmax.createNetwork({2, 4, 3}, {Activations::ReLU(), Activations::Softmax()});

    vector<vector<float>> mc_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<vector<float>> mc_targets = {{1,0,0}, {0,1,0}, {0,0,1}, {0,1,0}};

    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (size_t i = 0; i < mc_inputs.size(); ++i) {
            mlp_softmax.train(mc_inputs[i], mc_targets[i]);
        }
    }

    cout << "\nMulticlass Results:\n";
    for (const auto& input : mc_inputs) {
        mlp_softmax.setInput(input);
        auto output = mlp_softmax.forwardPropagate();
        cout << input[0] << "," << input[1] << " => ";
        for (auto val : output) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}