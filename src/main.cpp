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
#include <stdexcept>

namespace fs = std::filesystem;
using namespace std;

// ==================== Funciones auxiliares ====================

void cargarDatos(const std::string& filename, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets) {
    std::ifstream archivo(filename);
    if (!archivo.is_open()) {
        std::ostringstream oss;
        oss << "Error: no se pudo abrir el archivo '" << filename << "'";
        throw std::runtime_error(oss.str());
    }

    std::string linea;
    while (std::getline(archivo, linea)) {
        std::istringstream ss(linea);
        float x1, x2, y;
        if (!(ss >> x1 >> x2 >> y)) {
            throw std::runtime_error("Error al leer una línea del archivo '" + filename + "'");
        }
        inputs.push_back({x1, x2});
        targets.push_back({y});
    }
}

vector<int> parseEstructuraRed(const string& estructura) {
    vector<int> red;
    stringstream ss(estructura);
    string token;
    while (getline(ss, token, ',')) {
        red.push_back(stoi(token));
    }
    return red;
}

ActivationFunction getActivationFunction(const string& name) {
    if (name == "sigmoid") return Activations::Sigmoid();
    if (name == "relu")    return Activations::ReLU();
    if (name == "tanh")    return Activations::Tanh();
    if (name == "softmax") return Activations::Softmax();
    throw invalid_argument("Función de activación no soportada: " + name);
}

vector<ActivationFunction> parseActivaciones(const string& s) {
    vector<ActivationFunction> funciones;
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        funciones.push_back(getActivationFunction(token));
    }
    return funciones;
}

vector<int> estructuraPorDefecto(const string& dataset) {
    if (dataset == "xor" || dataset == "and" || dataset == "or") return {2, 2, 1};
    if (dataset == "mnist") return {784, 128, 64, 10};
    throw invalid_argument("Dataset no reconocido para estructura por defecto.");
}

vector<ActivationFunction> funcionesPorDefecto(size_t capas, const string& dataset) {
    if (dataset == "xor" || dataset == "and" || dataset == "or") {
        return vector<ActivationFunction>(capas - 1, Activations::Sigmoid());
    }
    if (dataset == "mnist") {
        vector<ActivationFunction> activaciones;
        for (size_t i = 0; i < capas - 2; ++i)
            activaciones.push_back(Activations::ReLU());
        activaciones.push_back(Activations::Softmax());
        return activaciones;
    }
    throw invalid_argument("Dataset no reconocido para funciones por defecto.");
}

// ==================== MAIN ====================

int main(int argc, char* argv[]) {
    try {
        cout << "\n=========================================================\n";
        cout << "        Red Neuronal Multicapa - Ejecución Principal\n";
        cout << "=========================================================\n";

        if (argc < 2) {
            cerr << "Uso: " << argv[0] << " <modelo.bin> <dataset> [\"<estructura>\"] [\"<activaciones>\"] [epocas]\n";
            cerr << "Ejemplo: " << argv[0] << " modelo.bin xor \"2,2,1\" \"sigmoid,sigmoid\" 5000\n";
            cerr << "         " << argv[0] << " modelo.bin mnist \"784,128,64,10\" \"relu,relu,softmax\" 20\n";
            return 1;
        }

        string modelo = "modelo.bin";
        string dataset = "xor";
        vector<int> estructura;
        vector<ActivationFunction> activaciones;
        int epocas;

        // Argumentos por línea de comandos
        if (argc >= 2) modelo = argv[1];
        if (argc >= 3) dataset = argv[2];
        estructura = (argc >= 4) ? parseEstructuraRed(argv[3]) : estructuraPorDefecto(dataset);
        activaciones = (argc >= 5) ? parseActivaciones(argv[4]) : funcionesPorDefecto(estructura.size(), dataset);
        epocas = (argc >= 6) ? stoi(argv[5]) : (dataset == "mnist" ? 20 : 3000);

        cout << "\nModelo: " << modelo;
        cout << "\nDataset: " << dataset;
        cout << "\nEstructura: ";
        for (auto n : estructura) cout << n << " ";
        cout << "\nActivaciones: ";
        for (size_t i = 0; i < activaciones.size(); ++i) cout << "[" << i << "] ";
        cout << "\nÉpocas: " << epocas << endl;

        MultilayerPerceptron mlp;

        string modelo_path = "models/" + modelo;
        bool usarModeloExistente = false;

        if (fs::exists(modelo_path)) {
            cout << "\n[INFO] El modelo '" << modelo_path << "' ya existe.\n";

    
           
            cout << "¿Qué deseas hacer?\n";
            cout << "  [C]argar modelo existente\n";
            cout << "  [N]uevo modelo desde cero (sobrescribir)\n";
            cout << "  [X] Cancelar ejecución\n";
            cout << "Tu elección (C/N/X): ";
            char eleccion;
            cin >> eleccion;

            if (eleccion == 'C' || eleccion == 'c') {
                cout << "\n[INFO] Cargando modelo desde disco...\n";
                 // Crea la red antes de cargar pesos
               // mlp.createNetwork(estructura, activaciones);
                mlp.loadModel(modelo_path);
                usarModeloExistente = true;
            } else if (eleccion == 'N' || eleccion == 'n') {
                cout << "\n[INFO] Creando nuevo modelo...\n";
                mlp.createNetwork(estructura, activaciones);
            } else {
                cout << "\n[INFO] Ejecución cancelada por el usuario.\n";
                return 0;
            }
        } else {
            cout << "\n[INFO] No existe el modelo. Creando nuevo modelo...\n";
            mlp.createNetwork(estructura, activaciones);
        }

        if (usarModeloExistente) {
            cout << "\n[INFO] ¿Deseas usar el modelo existente solo para inferencia? [s/n]: ";
            char soloTest;
            cin >> soloTest;
            if (soloTest == 's' || soloTest == 'S') {
                cout << "\n[INFO] Saltando entrenamiento. Ejecutando test...\n";

                if (dataset == "mnist") {
                    auto test_images = MNISTDataset::loadImages("dataset/mnist/t10k-images.idx3-ubyte", 100);
                    auto test_labels = MNISTDataset::loadLabels("dataset/mnist/t10k-labels.idx1-ubyte", 100);

                    float accuracy = mlp.calculateAccuracy(test_images, test_labels);
                    cout << "\nPrecisión en test: " << accuracy << "%" << endl;

                    mlp.testModel(test_images, test_labels, true);
                } else {
                    vector<vector<float>> inputs, targets;
                    cargarDatos("dataset/" + dataset + "_test.txt", inputs, targets);

                    for (size_t i = 0; i < inputs.size(); ++i) {
                        mlp.setInput(inputs[i]);
                        auto output = mlp.forwardPropagate();
                        cout << inputs[i][0] << " " << dataset << " " << inputs[i][1] << " = " << output[0] << endl;
                    }
                }
                return 0;
            }
        }

        if (dataset == "xor" || dataset == "and" || dataset == "or") {
            cout << "\n--- Cargando datos del dataset " << dataset << " ---\n";
            vector<vector<float>> inputs, targets;
            cargarDatos("dataset/" + dataset + "_test.txt", inputs, targets);

            cout << "\n--- Entrenando red ---\n";
            mlp.trainDataset(inputs, targets, epocas);

            cout << "\n--- Resultados finales (" << dataset << ") ---\n";
            for (size_t i = 0; i < inputs.size(); ++i) {
                mlp.setInput(inputs[i]);
                auto output = mlp.forwardPropagate();
                cout << inputs[i][0] << " " << dataset << " " << inputs[i][1] << " = " << output[0] << endl;
            }

            cout << "\n[INFO] Guardando modelo entrenado en: " << modelo_path << "\n";
            mlp.saveModel(modelo_path);
        }
        else if (dataset == "mnist") {
            cout << "\n--- Cargando dataset MNIST ---\n";
            auto train_images = MNISTDataset::loadImages("dataset/mnist/train-images.idx3-ubyte", 1000);
            auto train_labels = MNISTDataset::loadLabels("dataset/mnist/train-labels.idx1-ubyte", 1000);
            auto test_images = MNISTDataset::loadImages("dataset/mnist/t10k-images.idx3-ubyte", 100);
            auto test_labels = MNISTDataset::loadLabels("dataset/mnist/t10k-labels.idx1-ubyte", 100);

            cout << "\n--- Entrenando red MNIST ---\n";
            mlp.trainDataset(train_images, train_labels, epocas);

            float accuracy = mlp.calculateAccuracy(test_images, test_labels);
            cout << "\nPrecisión en test: " << accuracy << "%" << endl;

            mlp.testModel(test_images, test_labels, true);

            cout << "\n[INFO] Guardando modelo entrenado en: " << modelo_path << "\n";
            mlp.saveModel(modelo_path);
        }
        else {
            throw invalid_argument("Dataset no reconocido: " + dataset);
        }

        cout << "\n=========================================================\n";
        cout << "            Proceso finalizado correctamente\n";
        cout << "=========================================================\n";

    } catch (const exception& e) {
        cerr << "\n[ERROR] " << e.what() << "\n";
        cerr << "Uso: " << argv[0] << " <modelo.bin> <dataset> [\"<estructura\"] [\"<activaciones\"] [epocas]\n";
        cerr << "Ejemplo: " << argv[0] << " modelo.bin xor \"2,2,1\" \"sigmoid,sigmoid\" 5000\n";
        cerr << "         " << argv[0] << " modelo.bin mnist \"784,128,64,10\" \"relu,relu,softmax\" 20\n";
        return 1;
    }

    return 0;
}
