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

namespace fs = std::filesystem;

#define RANDOM_SEED 42
using namespace std;

// Lee imágenes MNIST normalizadas
vector<vector<float>> leerImagenesMNIST(const string& archivo, int cantidad) {
    ifstream f(archivo, ios::binary);
    if (!f) throw runtime_error("No se pudo abrir archivo de imágenes");

    int32_t magic, num, rows, cols;
    f.read((char*)&magic, 4);
    f.read((char*)&num, 4);
    f.read((char*)&rows, 4);
    f.read((char*)&cols, 4);

    magic = __builtin_bswap32(magic);
    num   = __builtin_bswap32(num);
    rows  = __builtin_bswap32(rows);
    cols  = __builtin_bswap32(cols);

    vector<vector<float>> imagenes;
    for (int i = 0; i < cantidad; ++i) {
        vector<float> imagen(rows * cols);
        for (int j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            f.read((char*)&pixel, 1);
            imagen[j] = pixel / 255.0f;
        }
        imagenes.push_back(imagen);
    }
    return imagenes;
}

// Lee etiquetas MNIST y las convierte a one-hot
vector<vector<float>> leerEtiquetasMNIST(const string& archivo, int cantidad) {
    ifstream f(archivo, ios::binary);
    if (!f) throw runtime_error("No se pudo abrir archivo de etiquetas");

    int32_t magic, num;
    f.read((char*)&magic, 4);
    f.read((char*)&num, 4);

    magic = __builtin_bswap32(magic);
    num   = __builtin_bswap32(num);

    vector<vector<float>> etiquetas;
    for (int i = 0; i < cantidad; ++i) {
        uint8_t label;
        f.read((char*)&label, 1);
        vector<float> one_hot(10, 0.0f);
        one_hot[label] = 1.0f;
        etiquetas.push_back(one_hot);
    }
    return etiquetas;
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

int main(int argc, char* argv[]) {
    string modelo = "modelo_mnist.bin";
    vector<int> estructuraRed = {784, 128, 64, 10};
    int epocas = 10;

    if (argc > 1) modelo = argv[1];
    if (argc > 2) estructuraRed = parseEstructuraRed(argv[2]);
    if (argc > 3) epocas = stoi(argv[3]);

    MultiLayerPerceptron red( estructuraRed, Activations::Sigmoid(), 0.1f  );

    if (fs::exists("../models/" + modelo)) {
        cout << "Cargando modelo desde: models/" << modelo << endl;
        red.loadWeights(modelo);
    } else {
        cout << "Entrenando modelo y guardando en: models/" << modelo << endl;

        const int cantidad = 1000;
        auto imagenes = leerImagenesMNIST("../dataset/mnist/train-images.idx3-ubyte", cantidad);
        auto etiquetas = leerEtiquetasMNIST("../dataset/mnist/train-labels.idx1-ubyte", cantidad);

        red.trainDataset(imagenes, etiquetas, epocas);
        red.saveWeights("../models/" + modelo);
    }

    auto imagenes_test = leerImagenesMNIST("../dataset/mnist/t10k-images.idx3-ubyte", 10);
    auto etiquetas_test = leerEtiquetasMNIST("../dataset/mnist/t10k-labels.idx1-ubyte", 10);
    red.testDataset(imagenes_test, etiquetas_test);
    
    return 0;
}
