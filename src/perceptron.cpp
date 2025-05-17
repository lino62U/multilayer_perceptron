#include "perceptron.hpp"
#include <iostream>
#include <Activations.hpp>
#include <fstream>
#include <filesystem>


#define LEARNING_RATE 0.1f

MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<int>& structure, const ActivationFunction & activationFunction, float learningRate) {
    perceptron.buildNetwork(structure, activationFunction, learningRate);
}


std::vector<float> MultiLayerPerceptron::forward() {

    auto& layers = perceptron.getLayers();

    for (size_t i = 1; i < layers.size(); ++i) {
        for (auto& neurona : layers[i]) {
            neurona->forward();
        }
    }

    std::vector<float> salidas;
    for (auto& neurona : layers.back()) {
        if (!neurona->getIsBias())
            salidas.push_back(neurona->getOutput());
    }
    return salidas;

}

void MultiLayerPerceptron::backpropagation(const std::vector<float>& targets) {
    auto& layers = perceptron.getLayers();

     // Capa de salida
    for (size_t i = 0; i < layers.back().size(); ++i) {
        if (!layers.back()[i]->getIsBias())
            layers.back()[i]->computeOutputDelta(targets[i]);
    }

    // Capas ocultas hacia atrás
    for (int i = layers.size() - 2; i > 0; --i) {
        for (auto& neurona : layers[i]) {
            if (!neurona->getIsBias())
                neurona->computeHiddenDelta();
        }
    }

    // Actualizar pesos
    for (size_t i = 1; i < layers.size(); ++i) {
        for (auto& neurona : layers[i]) {
            neurona->updateWeights();
        }
    }

   
}


void MultiLayerPerceptron::setInput(const std::vector<float>& inputs) {
    auto& inputLayer = perceptron.getLayers()[0];
    for (int i = 0; i < inputs.size(); ++i) {
        inputLayer[i]->setOutput( inputs[i] );
    }
}

void MultiLayerPerceptron::printNetwork() {
    auto& layers = perceptron.getLayers();
    for (int i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i << ":\n";
        for (auto& neuron : layers[i]) {
            std::cout << "  Output: " << neuron->getOutput();
            if (neuron->getIsBias()) std::cout << " [bias]";
            std::cout << std::endl;
        }
    }
}

void MultiLayerPerceptron::printFinalWeights() {
    auto& layers = perceptron.getLayers();

    for (int layerIdx = 1; layerIdx < layers.size(); ++layerIdx) {
        std::cout << "Layer " << layerIdx << ":\n";
        for (int neuronIdx = 0; neuronIdx < layers[layerIdx].size(); ++neuronIdx) {
            Neuron* neuron = layers[layerIdx][neuronIdx];
            if (neuron->getIsBias()) {
                std::cout << "  Neuron " << neuronIdx << " is BIAS\n";
                continue;
            }
            std::cout << "  Neuron " << neuronIdx << ":\n";
            int inputIdx = 0;
            for (auto& conn : neuron->getNext()) {
                std::string fromName = conn.first->getIsBias() ? "bias" : std::string(1, 'A' + inputIdx++);
                std::cout << "    From " << fromName << ": weight = " << *conn.second << "\n";
            }
        }
    }
}


void MultiLayerPerceptron::saveWeights(const std::string& filename) {
    // Crea la carpeta 'models' si no existe
    std::filesystem::create_directories("models");

    std::ofstream file("../models/" + filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for saving weights.\n";
        return;
    }

    const auto& network = perceptron.getLayers();
    for (size_t i = 1; i < network.size(); ++i) {
        for (const auto& neuron : network[i]) {
            for (const auto& input : neuron->getPrev()) {
                float* weight = input.second;
                file.write(reinterpret_cast<char*>(weight), sizeof(float));
            }
        }
    }
    file.close();
    std::cout << "Weights saved to models/" << filename << "\n";
}

void MultiLayerPerceptron::loadWeights(const std::string& filename) {
    std::ifstream file("../models/" + filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for loading weights.\n";
        return;
    }

    auto& network = perceptron.getLayers();
    for (size_t i = 1; i < network.size(); ++i) {
        for (const auto& neuron : network[i]) {
            for (const auto& input : neuron->getPrev()) {
                float* weight = input.second;
                file.read(reinterpret_cast<char*>(weight), sizeof(float));
            }
        }
    }
    file.close();
    std::cout << "Weights loaded from models/" << filename << "\n";
}


float MultiLayerPerceptron::calculateLoss(const std::vector<float>& output,
                                          const std::vector<float>& target) const {
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i)
        loss += 0.5f * std::pow(output[i] - target[i], 2);
    return loss;
}

float MultiLayerPerceptron::evaluateAccuracy(const std::vector<std::vector<float>>& inputs,
                                             const std::vector<std::vector<float>>& targets) const {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        const auto& target = targets[i];

        const_cast<MultiLayerPerceptron*>(this)->setInput(input);  // for calling forward()
        auto output = const_cast<MultiLayerPerceptron*>(this)->forward();

        int pred = std::max_element(output.begin(), output.end()) - output.begin();
        int real = std::max_element(target.begin(), target.end()) - target.begin();
        if (pred == real) ++correct;
    }
    return (correct * 100.0f) / inputs.size();
}

void MultiLayerPerceptron::trainDataset(const std::vector<std::vector<float>>& inputs,
                                        const std::vector<std::vector<float>>& targets,
                                        int epochs) {
    for (int epoca = 0; epoca < epochs; ++epoca) {
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            setInput(inputs[i]);
            auto output = forward();
            total_loss += calculateLoss(output, targets[i]);

            int pred = std::max_element(output.begin(), output.end()) - output.begin();
            int real = std::max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();
            if (pred == real) ++correct;

            backpropagation(targets[i]);
        }

        std::cout << "Época " << epoca + 1
                  << " | Accuracy: " << (correct * 100.0 / inputs.size())
                  << "% | Loss: " << total_loss / inputs.size() << "\n";
    }
}

void MultiLayerPerceptron::mostrarImagenConsola(const std::vector<float>& imagen) const {
    const std::string escala = " .:-=+*#%@";
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            float pixel = imagen[i * 28 + j];
            int nivel = static_cast<int>(pixel * (escala.size() - 1));
            std::cout << escala[nivel];
        }
        std::cout << '\n';
    }
}

void MultiLayerPerceptron::testDataset(const std::vector<std::vector<float>>& inputs,
                                       const std::vector<std::vector<float>>& targets,
                                       bool showImages) const {
    int correct = 0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        const_cast<MultiLayerPerceptron*>(this)->setInput(inputs[i]);
        auto output = const_cast<MultiLayerPerceptron*>(this)->forward();

        int pred = std::max_element(output.begin(), output.end()) - output.begin();
        int real = std::max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();

        if (showImages) {
            std::cout << "Imagen #" << i + 1 << "\n";
            mostrarImagenConsola(inputs[i]);
            std::cout << "Etiqueta real: " << real << " | Predicción: " << pred << "\n\n";
        }

        if (pred == real) ++correct;
    }

    std::cout << "Precisión total: " << (correct * 100.0 / inputs.size()) << "%\n";
}


