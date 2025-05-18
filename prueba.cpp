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

using ActivationFunc = float(*)(float);

struct ActivationFunction {
    ActivationFunc activate;
    ActivationFunc derive;

    ActivationFunction(ActivationFunc a = nullptr, ActivationFunc d = nullptr)
        : activate(a), derive(d) {}
};

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
        return ActivationFunction(nullptr, nullptr); // Special case
    }
}

struct Neuron {
    float output = 0.0f;
    float net_input = 0.0f;
    float delta = 0.0f;
    bool is_bias = false;

    vector<pair<Neuron*, float*>> inputs;
    vector<pair<Neuron*, float*>> outputs;

    ActivationFunction activation;

    Neuron(ActivationFunction act = Activations::Sigmoid()) 
        : activation(act) {}

    void computeOutput() {
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
        // Softmax is handled at layer level
    }

    void computeDelta(bool is_output_neuron, float target = 0.0f) {
        if (is_output_neuron) {
            delta = (output - target); // For softmax, derivative is already considered
        } else {
            float sum = 0.0f;
            for (const auto& [neuron, weight] : outputs) {
                sum += *weight * neuron->delta;
            }
            delta = activation.derive(output) * sum;
        }
    }

    void updateWeights() {
        if (is_bias) return;
        for (auto& [neuron, weight] : inputs) {
            *weight -= LEARNING_RATE * delta * neuron->output;
        }
    }
};

class MultilayerPerceptron {
    vector<vector<Neuron*>> layers;
    vector<ActivationFunction> layer_activations;
    bool softmax_output = false;

    void connectLayers() {
        srand(RANDOM_SEED);
        for (size_t i = 1; i < layers.size(); ++i) {
            for (auto& neuron : layers[i]) {
                if (neuron->is_bias) continue;
                for (auto& prev_neuron : layers[i-1]) {
                    float* weight = new float((rand()%100)/100.0f - 0.5f);
                    neuron->inputs.emplace_back(prev_neuron, weight);
                    prev_neuron->outputs.emplace_back(neuron, weight);
                }
            }
        }
    }

    void applySoftmax() {
        vector<float> z;
        for (auto& neuron : layers.back()) {
            if (!neuron->is_bias) {
                z.push_back(neuron->net_input);
            }
        }

        auto softmax_values = Activations::softmax(z);
        
        size_t idx = 0;
        for (auto& neuron : layers.back()) {
            if (!neuron->is_bias) {
                neuron->output = softmax_values[idx++];
            }
        }
    }

public:
    ~MultilayerPerceptron() {
        for (auto& layer : layers) {
            for (auto& neuron : layer) {
                for (auto& [_, weight] : neuron->inputs) {
                    delete weight;
                }
                delete neuron;
            }
        }
    }

    void createNetwork(const vector<int>& architecture, 
                      const vector<ActivationFunction>& activations) {
        if (activations.size() != architecture.size() - 1) {
            throw runtime_error("Number of activation functions must match hidden layers + output");
        }
        
        layer_activations = activations;
        softmax_output = (activations.back().activate == nullptr);

        // Create layers
        for (size_t i = 0; i < architecture.size(); ++i) {
            vector<Neuron*> layer;
            int neurons_count = architecture[i] + (i != architecture.size()-1); // +1 for bias
            
            ActivationFunction act = (i == 0) ? Activations::Sigmoid() : layer_activations[i-1];
            
            for (int j = 0; j < neurons_count; ++j) {
                Neuron* n = new Neuron(act);
                if (j == neurons_count-1 && i != architecture.size()-1) {
                    n->is_bias = true;
                    n->output = 1.0f;
                }
                layer.push_back(n);
            }
            layers.push_back(layer);
        }
        connectLayers();
    }

    void setInput(const vector<float>& inputs) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            layers[0][i]->output = inputs[i];
        }
    }

    vector<float> forwardPropagate() {
        for (size_t i = 1; i < layers.size(); ++i) {
            for (auto& neuron : layers[i]) {
                neuron->computeOutput();
            }
        }

        if (softmax_output) {
            applySoftmax();
        }

        vector<float> output;
        for (auto& neuron : layers.back()) {
            if (!neuron->is_bias) {
                output.push_back(neuron->output);
            }
        }
        return output;
    }

    void backPropagate(const vector<float>& targets) {
        // Output layer
        for (size_t i = 0; i < layers.back().size(); ++i) {
            if (!layers.back()[i]->is_bias) {
                layers.back()[i]->computeDelta(true, targets[i]);
            }
        }

        // Hidden layers
        for (int i = layers.size()-2; i > 0; --i) {
            for (auto& neuron : layers[i]) {
                if (!neuron->is_bias) {
                    neuron->computeDelta(false);
                }
            }
        }

        // Update weights
        for (size_t i = 1; i < layers.size(); ++i) {
            for (auto& neuron : layers[i]) {
                neuron->updateWeights();
            }
        }
    }

    void train(const vector<float>& input, const vector<float>& target) {
        setInput(input);
        forwardPropagate();
        backPropagate(target);
    }

    void printNetwork() const {
        for (size_t i = 0; i < layers.size(); ++i) {
            cout << "Layer " << i << " (";
            if (i == 0) cout << "Input";
            else if (i == layers.size()-1) cout << (softmax_output ? "Softmax" : "Output");
            else cout << "Hidden";
            cout << "):\n";
            
            for (const auto& neuron : layers[i]) {
                cout << "  Output: " << neuron->output << ", Delta: " << neuron->delta;
                if (neuron->is_bias) cout << " [bias]";
                cout << endl;
            }
        }
    }
};

int main() {
    MultilayerPerceptron mlp;
    
    // Ejemplo XOR con softmax (aunque no es el caso ideal)
    cout << "XOR with Sigmoid/Tanh (single output):\n";
    mlp.createNetwork({2, 2, 1}, {Activations::Sigmoid(), Activations::Sigmoid()});

    vector<vector<float>> training_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<vector<float>> training_targets = {{0}, {1}, {1}, {0}};

    // Training
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