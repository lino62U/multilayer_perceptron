#include "perceptron.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>
#include <filesystem>
#include <cstdint>
#include <Activations.hpp>
#include <MNISTDataset.hpp>

namespace fs = std::filesystem;

#define RANDOM_SEED 42
using namespace std;


// ==================== EJEMPLOS DE USO ====================
int main() {
    // Ejemplo XOR
    cout << "Entrenando red XOR...\n";
    MultilayerPerceptron mlp;
    mlp.createNetwork({2, 2, 1}, {Activations::Sigmoid(), Activations::Sigmoid()});

    vector<vector<float>> xor_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<vector<float>> xor_targets = {{0}, {1}, {1}, {0}};

    mlp.trainDataset(xor_inputs, xor_targets, 5000);

    cout << "\nResultados XOR:\n";
    for (size_t i = 0; i < xor_inputs.size(); ++i) {
        mlp.setInput(xor_inputs[i]);
        auto output = mlp.forwardPropagate();
        cout << xor_inputs[i][0] << " XOR " << xor_inputs[i][1] << " = " << output[0] << endl;
    }

    // Ejemplo MNIST (reducido para demostración)
    cout << "\nCargando MNIST reducido...\n";
    try {
        auto train_images = MNISTDataset::loadImages("../dataset/mnist/train-images.idx3-ubyte", 1000);
        auto train_labels = MNISTDataset::loadLabels("../dataset/mnist/train-labels.idx1-ubyte", 1000);
        auto test_images = MNISTDataset::loadImages("../dataset/mnist/t10k-images.idx3-ubyte", 10);
        auto test_labels = MNISTDataset::loadLabels("../dataset/mnist/t10k-labels.idx1-ubyte", 10);

        cout << "\nMostrando primera imagen de entrenamiento:\n";
        MNISTDataset::displayImage(train_images[0]);
        cout << "Etiqueta: ";
        for (auto val : train_labels[0]) cout << val << " ";
        cout << endl;

        MultilayerPerceptron mnist_mlp;
        mnist_mlp.createNetwork({784, 128, 10}, {Activations::ReLU(), Activations::Softmax()});

        cout << "\nEntrenando red MNIST...\n";
        mnist_mlp.trainDataset(train_images, train_labels, 20);

        cout << "\nProbando red MNIST...\n";
        float accuracy = mnist_mlp.calculateAccuracy(test_images, test_labels);
        cout << "Precisión en test: " << accuracy << "%" << endl;

          cout << "TESTING IMAGES  " << endl;

        mnist_mlp.testModel(test_images, test_labels, true);





        // Guardar y cargar modelo
        cout << "\nGuardando modelo...\n";
        mnist_mlp.saveModel("mnist_model.bin");

        cout << "Cargando modelo...\n";
        MultilayerPerceptron loaded_mlp;
        loaded_mlp.loadModel("mnist_model.bin");

        cout << "Precisión del modelo cargado: " 
             << loaded_mlp.calculateAccuracy(test_images, test_labels) << "%" << endl;

    } catch (const exception& e) {
        cerr << "Error con MNIST: " << e.what() << endl;
        cerr << "Asegúrate de tener los archivos MNIST en el directorio actual." << endl;
    }

    return 0;
}