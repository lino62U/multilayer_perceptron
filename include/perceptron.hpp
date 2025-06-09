#pragma once

#include "layers.hpp"
#include <vector>
#include <string>  // ← AÑADE ESTO
#include <algorithm>


class MultilayerPerceptron {
    vector<Layer*> layers;
    bool softmax_output = false;

public:
    ~MultilayerPerceptron();
    void createNetwork(const vector<int>& architecture, 
                      const vector<ActivationFunction>& activations);
    void setInput(const vector<float>& inputs) const;
    vector<float> forwardPropagate() const;
    void backPropagate(const vector<float>& targets);
    void train(const vector<float>& input, const vector<float>& target);
    
    // Nuevas funciones de entrenamiento y evaluación
    void trainDataset(const vector<vector<float>>& inputs, 
                     const vector<vector<float>>& targets, 
                     int epochs, int batch_size = 1);
    float calculateLoss(const vector<float>& output, const vector<float>& target) const;
    float calculateAccuracy(const vector<vector<float>>& inputs, 
                          const vector<vector<float>>& targets) const;
    
    // Funciones para guardar/cargar modelo
    void saveModel(const string& filename) const;
    void loadModel(const string& filename);
    
    void printNetwork() const;

     void testModel(const vector<vector<float>>& test_images, 
                  const vector<vector<float>>& test_labels,
                  bool show_details = false) const {
        if (test_images.size() != test_labels.size()) {
            throw runtime_error("Test images and labels must have the same size");
        }

        int correct = 0;
        for (size_t i = 0; i < test_images.size(); ++i) {
            setInput(test_images[i]);
            auto output = forwardPropagate();

            int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
            int actual = distance(test_labels[i].begin(), max_element(test_labels[i].begin(), test_labels[i].end()));

            if (show_details) {
                cout << "Test Sample #" << i + 1 << "\n";
                displayImage(test_images[i]);
                cout << "Actual: " << actual << " | Predicted: " << predicted << "\n\n";
            }

            if (predicted == actual) correct++;
        }

        float accuracy = static_cast<float>(correct) / test_images.size() * 100.0f;
        cout << "Test Accuracy: " << accuracy << "% (" << correct << "/" << test_images.size() << ")\n";
    }

    static void displayImage(const vector<float>& image, int rows = 28, int cols = 28) {
        const string shades = " .:-=+*#%@";
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float pixel = image[i * cols + j];
                int level = static_cast<int>(pixel * (shades.size() - 1));
                cout << shades[level] << shades[level];
            }
            cout << endl;
        }
    }
};
