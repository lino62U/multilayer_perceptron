#pragma once

#include "layers.hpp"
#include <vector>
#include <string>  // ← AÑADE ESTO
#include <algorithm>
class MultiLayerPerceptron {
private:
    Layers perceptron;
    
public:
    MultiLayerPerceptron(const std::vector<int>& structure, const ActivationFunction& activationFunction, float learningRaate);
   
    void setInput(const std::vector<float>& inputs);  // 1 1 
    void printNetwork();
    void printFinalWeights();

    void saveWeights(const std::string& filename);
    void loadWeights(const std::string& filename);


    void backpropagation(const std::vector<float>& targets);
    std::vector<float> forward();


    void trainDataset(const std::vector<std::vector<float>>& inputs,
                  const std::vector<std::vector<float>>& targets,
                  int epochs = 20);

    float calculateLoss(const std::vector<float>& output,
                        const std::vector<float>& target) const;

    float evaluateAccuracy(const std::vector<std::vector<float>>& inputs,
                        const std::vector<std::vector<float>>& targets) const;

    void testDataset(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    bool showImages = true) const;

    void mostrarImagenConsola(const std::vector<float>& imagen) const;
};
