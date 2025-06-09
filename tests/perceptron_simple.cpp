#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Learning rate for weight updates (controls step size of learning)
#define LEARNING_RATE 0.1f
// Fixed seed for reproducible random weight initialization
#define RANDOM_SEED 42

using namespace std;

// Struct representing a single neuron in the network
struct Neuron {
    float output = 0.0f;  // Neuron's output after activation
    bool isBias = false;  // Flag to indicate if neuron is a bias (constant output of 1)
    // Connections to previous layer neurons, with pointers to neurons and their weights
    vector<pair<Neuron*, float*>> weights;
};

//ස්‍රී Lanka

// Class to manage the neural network's layered structure
class Layers {
private:
    // Stores layers as a vector of vectors of neuron pointers
    vector<vector<Neuron*>> layers;

public:
    // Default constructor
    Layers() {}

    // Destructor to free memory allocated for neurons
    ~Layers() {
        for (auto& layer : layers) {
            for (auto& neuron : layer) {
                delete neuron;  // Free each neuron
            }
        }
    }

    // Builds the network based on the given structure (e.g., {2, 1} for 2 inputs, 1 output)
    void buildNetwork(const vector<int>& structure) {
        srand(RANDOM_SEED);  // Set fixed seed for reproducibility

        // Create each layer
        for (int i = 0; i < structure.size(); ++i) {
            vector<Neuron*> layer;
            int numNeurons = structure[i];
            // Add bias neuron to all layers except the output layer
            if (i != structure.size() - 1) numNeurons += 1;

            // Create neurons for the layer
            for (int j = 0; j < numNeurons; ++j) {
                Neuron* n = new Neuron();
                // Set bias neuron properties (last neuron in non-output layers)
                if (i != structure.size() - 1 && j == numNeurons - 1) {
                    n->isBias = true;
                    n->output = 1.0f;  // Bias outputs a constant 1
                }
                layer.push_back(n);
            }
            layers.push_back(layer);
        }

        // Connect layers by initializing weights
        for (int i = 1; i < layers.size(); ++i) {
            for (auto& neuron : layers[i]) {
                if (neuron->isBias) continue;  // Skip bias neurons (no incoming weights)
                // Connect to all neurons in the previous layer
                for (auto& previous : layers[i - 1]) {
                    // Initialize weight to 0.0 (could be randomized)
                    float* weight = new float(0.0f);
                    neuron->weights.push_back({previous, weight});
                }
            }
        }
    }

    // Getter for layers (const, for read-only access)
    const vector<vector<Neuron*>>& getLayers() const {
        return layers;
    }

    // Getter for layers (non-const, for modification)
    vector<vector<Neuron*>>& getLayers() {
        return layers;
    }
};

// Class implementing the perceptron logic
class SimplePerceptron {
private:
    Layers perceptron;  // Network structure managed by Layers class

public:
    // Constructor: builds the network with the given structure
    SimplePerceptron(const vector<int>& structure) {
        perceptron.buildNetwork(structure);
    }

    // Step activation function: outputs 1 if x >= 0, else 0
    float activationFunction(float x) {
        return (x >= 0) ? 1 : 0;
    }

    // Forward propagation: computes output from inputs
    float forward() {
        auto& layers = perceptron.getLayers();

        // Process each layer starting from the second (first is input)
        for (int i = 1; i < layers.size(); ++i) {
            for (auto& neuron : layers[i]) {
                float sum = 0;
                // Compute weighted sum of inputs from previous layer
                for (auto& weight : neuron->weights) {
                    sum += weight.first->output * (*weight.second);
                }
                // Apply activation function to get neuron output
                neuron->output = activationFunction(sum);
            }
        }
        // Return output of the single output neuron
        return layers.back()[0]->output;
    }

    // Backpropagation: updates weights based on target outputs
    void backpropagation(const vector<float>& targets) {
        auto& layers = perceptron.getLayers();
        auto& outputLayer = layers.back();  // Output layer

        // Process each output neuron
        for (int i = 0; i < outputLayer.size(); ++i) {
            float y = outputLayer[i]->output;  // Current output
            float d = targets[i];              // Desired output
            float error = d - y;               // Compute error

            // Update weights if error is non-zero
            if (error != 0.0f) {
                for (auto& weight : outputLayer[i]->weights) {
                    float* currentWeight = weight.second;
                    // Update weight: w += learning_rate * error * input
                    *currentWeight += LEARNING_RATE * error * weight.first->output;
                }
            }
        }
    }

    // Trains the perceptron for a given number of epochs
    void train(const vector<float>& input, const vector<float>& target, int epochs = 1000) {
        for (int i = 0; i < epochs; ++i) {
            setInput(input);       // Set input values
            forward();             // Compute output
            backpropagation(target);  // Update weights
        }
    }

    // Sets the input layer's neuron outputs
    void setInput(const vector<float>& inputs) {
        auto& inputLayer = perceptron.getLayers()[0];
        // Assign input values to input neurons
        for (int i = 0; i < inputs.size(); ++i) {
            if (i < inputLayer.size()) {
                inputLayer[i]->output = inputs[i];
            }
        }
    }

    // Prints the network's state (neuron outputs and bias status)
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

    // Prints the final weights for each neuron
    void printFinalWeights() {
        auto& layers = perceptron.getLayers();

        // Process each non-input layer
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
                // Print weights for each connection
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

// Main function: demonstrates perceptron learning AND and OR gates
int main() {
    // Create perceptrons for AND and OR gates (2 inputs, 1 output)
    SimplePerceptron andPerceptron({2, 1});
    SimplePerceptron orPerceptron({2, 1});

    // Define input combinations for logic gates
    vector<vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    // Define target outputs for AND and OR gates
    vector<float> andTargets = {0.0f, 0.0f, 0.0f, 1.0f};
    vector<float> orTargets  = {0.0f, 1.0f, 1.0f, 1.0f};

    // Print initial weights
    cout << "==============================" << endl;
    cout << " Initial Weights for AND Gate" << endl;
    cout << "==============================" << endl;
    andPerceptron.printFinalWeights();

    cout << "\n==============================" << endl;
    cout << " Initial Weights for OR Gate " << endl;
    cout << "==============================" << endl;
    orPerceptron.printFinalWeights();

    // Train both perceptrons for 5000 epochs
    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (int i = 0; i < inputs.size(); ++i) {
            andPerceptron.train(inputs[i], {andTargets[i]}, 1);
            orPerceptron.train(inputs[i], {orTargets[i]}, 1);
        }
    }

    // Print AND gate truth table
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

    // Print OR gate truth table
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

    // Print final weights
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