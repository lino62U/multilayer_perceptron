#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42  // Fixed seed

using namespace std;

struct Neuron {
    float output = 0.0f;
    bool isBias = false;
    vector<pair<Neuron*, float*>> weights;  // connection to the previous layer
};

class Layers {
private:
    vector<vector<Neuron*>> layers;

public:
    // Default constructor
    Layers() {}

    // Destructor to free memory
    ~Layers() {
        for (auto& layer : layers) {
            for (auto& neuron : layer) {
                delete neuron;
            }
        }
    }

    // Method to construct the network with the desired structure
    void buildNetwork(const vector<int>& structure) {
        srand(RANDOM_SEED);  // Fixed seed

        for (int i = 0; i < structure.size(); ++i) {
            vector<Neuron*> layer;
            int numNeurons = structure[i];
            if (i != structure.size() - 1) numNeurons += 1;  // Add bias if not the output layer

            for (int j = 0; j < numNeurons; ++j) {
                Neuron* n = new Neuron();

                if (i != structure.size() - 1 && j == numNeurons - 1) {
                    n->isBias = true;
                    n->output = 1.0f;  // Bias has a fixed output of 1
                }
                layer.push_back(n);
            }

            layers.push_back(layer);
        }

        // Connect layers
        for (int i = 1; i < layers.size(); ++i) {
            for (auto& neuron : layers[i]) {
                if (neuron->isBias) continue;
                for (auto& previous : layers[i - 1]) {
                    //float* weight = new float(((rand() % 100) / 100.0f) - 0.5f); // [-0.5, 0.5]
                    float* weight = new float(0.0f);

                    neuron->weights.push_back({previous, weight});
                }
            }
        }
    }

    // Getter for layers (const to prevent modification)
    const vector<vector<Neuron*>>& getLayers() const {
        return layers;
    }

    // Getter (non-const) in case modification is needed
    vector<vector<Neuron*>>& getLayers() {
        return layers;
    }
};

class SimplePerceptron {
private:
    Layers perceptron;

public:
    // Constructor that takes the network structure and builds the network
    SimplePerceptron(const vector<int>& structure) {
        perceptron.buildNetwork(structure);
    }

    // Activation function (step function)
    float activationFunction(float x) {
        return (x >= 0) ? 1 : 0;
    }

    // Forward propagation
    float forward() {
        auto& layers = perceptron.getLayers();

        // Propagate forward (each layer)
        for (int i = 1; i < layers.size(); ++i) {  // Start from the second layer
            for (auto& neuron : layers[i]) {
                float sum = 0;

                // Sum outputs of the previous layer neurons multiplied by their weights
                for (auto& weight : neuron->weights) {
                    sum += weight.first->output * (*weight.second);
                }

                // Apply the activation function (step function)
                neuron->output = activationFunction(sum);
            }
        }

        return layers.back()[0]->output;  // Return the network output
    }

    // Backpropagation (weight update)
    void backpropagation(const vector<float>& targets) {
        auto& layers = perceptron.getLayers();
        auto& outputLayer = layers.back();  // Last layer (output layer)

        float error = 0.0f;

        for (int i = 0; i < outputLayer.size(); ++i) {
            float y = outputLayer[i]->output;   // Current output of the network
            float d = targets[i];        // Target value

            float error = d - y;           // Error of the output neuron
           

            // Update weights only if the error is non-zero
            if (error != 0.0f) {
                for (auto& weight : outputLayer[i]->weights) {
                    float* currentWeight = weight.second;
                    *currentWeight += LEARNING_RATE * error * weight.first->output;
                }
            }
        }

       // cout << "Total error: " << error << endl;
    }

    // Training for multiple epochs
    void train(const vector<float>& input, const vector<float>& target, int epochs = 1000) {
        for (int i = 0; i < epochs; ++i) {
            setInput(input);           // Set inputs
            forward();                   // Run the network
            backpropagation(target);    // Apply backpropagation
        }
    }

    // Set the inputs
    void setInput(const vector<float>& inputs) {
        auto& inputLayer = perceptron.getLayers()[0];

        for (int i = 0; i < inputs.size(); ++i) {
            if (i < inputLayer.size()) {
                inputLayer[i]->output = inputs[i];
            }
        }
    }

    // Method to print the network state (outputs and bias)
    void printNetwork() {
        auto& layers = perceptron.getLayers();
        for (int i = 0; i < layers.size(); ++i) {
            cout << "Layer " << i << ":\n";
            for (auto& neuron : layers[i]) {
                cout << "  Output: " << neuron->output;
                if (neuron->isBias) cout << " [bias]";
                cout << endl;
            }
        }
    }
    void printFinalWeights() {
        auto& layers = perceptron.getLayers();

        for (int layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
            cout << "Layer " << layerIdx << ":\n";
            for (int neuronIdx = 0; neuronIdx < layers[layerIdx].size(); ++neuronIdx) {
                Neuron* neuron = layers[layerIdx][neuronIdx];
                if (neuron->isBias) {
                    cout << "  Neuron " << neuronIdx << " is BIAS (output = 1.0)\n";
                    continue;
                }

                cout << "  Neuron " << neuronIdx << ":\n";

                int inputIdx = 0;
                for (auto& conn : neuron->weights) {
                    Neuron* from = conn.first;
                    float* weight = conn.second;

                    string fromName;
                    if (from->isBias) {
                        fromName = "bias";
                    } else {
                        fromName = (inputIdx == 0) ? "A" : "B";
                        inputIdx++;
                    }

                    cout << "    From " << fromName << ": weight = " << *weight << "\n";
                }
            }
        }
    }

};
int main() {
    SimplePerceptron andPerceptron({2, 1});
    SimplePerceptron orPerceptron({2, 1});

    vector<vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    vector<float> andTargets = {0.0f, 0.0f, 0.0f, 1.0f};
    vector<float> orTargets  = {0.0f, 1.0f, 1.0f, 1.0f};

    // Pesos iniciales
    cout << "==============================" << endl;
    cout << " Initial Weights for AND Gate" << endl;
    cout << "==============================" << endl;
    andPerceptron.printFinalWeights();

    cout << "\n==============================" << endl;
    cout << " Initial Weights for OR Gate " << endl;
    cout << "==============================" << endl;
    orPerceptron.printFinalWeights();

    // Entrenamiento
    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (int i = 0; i < inputs.size(); ++i) {
            andPerceptron.train(inputs[i], {andTargets[i]}, 1);
            orPerceptron.train(inputs[i], {orTargets[i]}, 1);
        }
    }

    // Resultados AND
    cout << "\n===================" << endl;
    cout << " Truth Table: AND  " << endl;
    cout << "===================" << endl;
    cout << " A | B | Output" << endl;
    cout << "---------------" << endl;
    for (const auto& in : inputs) {
        andPerceptron.setInput(in);
        float out = andPerceptron.forward();
        cout << " " << in[0] << " | " << in[1] << " |   " << out << endl;
    }

    // Resultados OR
    cout << "\n===================" << endl;
    cout << " Truth Table: OR   " << endl;
    cout << "===================" << endl;
    cout << " A | B | Output" << endl;
    cout << "---------------" << endl;
    for (const auto& in : inputs) {
        orPerceptron.setInput(in);
        float out = orPerceptron.forward();
        cout << " " << in[0] << " | " << in[1] << " |   " << out << endl;
    }

    // Pesos finales
    cout << "\n==============================" << endl;
    cout << " Final Weights for AND Gate  " << endl;
    cout << "==============================" << endl;
    andPerceptron.printFinalWeights();

    cout << "\n==============================" << endl;
    cout << " Final Weights for OR Gate   " << endl;
    cout << "==============================" << endl;
    orPerceptron.printFinalWeights();

    return 0;
}
