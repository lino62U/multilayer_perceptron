#include "perceptron.hpp"
#include <iostream>
#include <Activations.hpp>
#include <fstream>
#include <filesystem>


#define LEARNING_RATE 0.1f

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

void MultilayerPerceptron::setInput(const vector<float>& inputs) const{
    layers[0]->setInputs(inputs);
}

vector<float> MultilayerPerceptron::forwardPropagate() const{
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

void MultilayerPerceptron::trainDataset(const vector<vector<float>>& inputs, 
                                       const vector<vector<float>>& targets, 
                                       int epochs, int batch_size) {
    if (inputs.size() != targets.size()) {
        throw runtime_error("Inputs and targets must have the same size");
    }

    // Determina cada cuántas épocas imprimir, máximo 20 líneas
    int max_lines = 20;
    int print_interval = max(1, epochs / max_lines);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            setInput(inputs[i]);
            auto output = forwardPropagate();
            backPropagate(targets[i]);

            // Calcular pérdida y precisión
            total_loss += calculateLoss(output, targets[i]);
            
            // Para precisión
            int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
            int actual = distance(targets[i].begin(), max_element(targets[i].begin(), targets[i].end()));
            if (predicted == actual) correct++;
        }

        // Mostrar solo en intervalos
        if ((epoch + 1) % print_interval == 0 || epoch == epochs - 1 || epoch == 0) {
            float avg_loss = total_loss / inputs.size();
            float accuracy = static_cast<float>(correct) / inputs.size() * 100.0f;

            cout << "Epoch " << epoch + 1 << "/" << epochs 
                 << " - Loss: " << avg_loss 
                 << " - Accuracy: " << accuracy << "%" << endl;
        }
    }
}


float MultilayerPerceptron::calculateLoss(const vector<float>& output, const vector<float>& target) const {
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        loss += 0.5f * pow(output[i] - target[i], 2); // MSE
    }
    return loss;
}

float MultilayerPerceptron::calculateAccuracy(const vector<vector<float>>& inputs, 
                                            const vector<vector<float>>& targets) const {
    if (inputs.size() != targets.size()) {
        throw runtime_error("Inputs and targets must have the same size");
    }

    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        setInput(inputs[i]);
        auto output = forwardPropagate();
        
        int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
        int actual = distance(targets[i].begin(), max_element(targets[i].begin(), targets[i].end()));
        
        if (predicted == actual) correct++;
    }

    return static_cast<float>(correct) / inputs.size() * 100.0f;
}

void MultilayerPerceptron::saveModel(const string& filename) const {
    ofstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    // Guardar arquitectura (número de neuronas sin bias)
    vector<int> architecture;
    for (size_t i = 0; i < layers.size(); ++i) {
        int neurons = layers[i]->size();
        if (i != layers.size()-1) neurons--; // Excluir bias en capas que no son salida
        architecture.push_back(neurons);
    }

    size_t num_layers = architecture.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(architecture.data()), num_layers * sizeof(int));

    // Guardar IDs de funciones de activación (para capas ocultas y salida)
    // Nota: la capa 0 es input y no tiene activación
    for (size_t i = 1; i < layers.size(); ++i) {
        int act_id = getActivationId(layers[i]->activation);
        if (act_id == -1) throw runtime_error("Unknown activation function when saving model");
        file.write(reinterpret_cast<const char*>(&act_id), sizeof(int));
    }

    // Guardar pesos
    for (size_t i = 1; i < layers.size(); ++i) {
        layers[i]->saveWeights(file);
    }
}


void MultilayerPerceptron::loadModel(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Cannot open file for reading: " + filename);
    }

    // Leer arquitectura
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
    
    vector<int> architecture(num_layers);
    file.read(reinterpret_cast<char*>(architecture.data()), num_layers * sizeof(int));

    // Leer activaciones
    vector<ActivationFunction> activations;
    for (size_t i = 1; i < num_layers; ++i) {  // empiezas en 1 porque input no tiene activación
        int act_id;
        file.read(reinterpret_cast<char*>(&act_id), sizeof(int));
        activations.push_back(getActivationById(act_id));
    }

    // Recrear la red con arquitectura y activaciones
    createNetwork(architecture, activations);

    // Cargar pesos
    for (size_t i = 1; i < layers.size(); ++i) {
        layers[i]->loadWeights(file);
    }
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
