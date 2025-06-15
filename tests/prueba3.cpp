#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;
namespace fs = std::filesystem;

enum class OptimizerType {
    SGD,    // Descenso de gradiente estocástico (por defecto)
    RMSProp,
    Adam
};

// ==================== DECLARACIONES ====================
using ActivationFunc = float(*)(float);

struct ActivationFunction {
    ActivationFunc activate;
    ActivationFunc derive;

    ActivationFunction(ActivationFunc a = nullptr, ActivationFunc d = nullptr)
        : activate(a), derive(d) {}
};

namespace Activations {
    // Funciones de activación
    static float sigmoid(float x);
    static float sigmoid_derivative(float y);
    static float relu(float x);
    static float relu_derivative(float x);
    static float tanh_act(float x);
    static float tanh_derivative(float y);
    static vector<float> softmax(const vector<float>& z);
    
    // Funciones de construcción
    static ActivationFunction Sigmoid();
    static ActivationFunction ReLU();
    static ActivationFunction Tanh();
    static ActivationFunction Softmax();
}

// Modificar la estructura Neuron para incluir parámetros de optimización
struct Neuron {
    float output = 0.0f;
    float net_input = 0.0f;
    float delta = 0.0f;
    bool is_bias = false;
    vector<pair<Neuron*, float*>> inputs;
    vector<pair<Neuron*, float*>> outputs;
    ActivationFunction activation;
    
    // Parámetros para optimizadores
    OptimizerType optimizer = OptimizerType::SGD;
    float learning_rate = LEARNING_RATE;
    
    // Para RMSProp
    vector<float> squared_gradients;
    float rmsprop_rho = 0.9f;
    float rmsprop_epsilon = 1e-8f;
    
    // Para Adam
    vector<float> m;  // Primer momento (media)
    vector<float> v;  // Segundo momento (varianza no centrada)
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1e-8f;
    int t = 0;        // Contador de pasos

    Neuron(ActivationFunction act = Activations::Sigmoid());
    void computeOutput();
    void computeDelta(bool is_output_neuron, float target = 0.0f);
    void updateWeights();
    
    // Nuevas funciones para optimizadores
    void initializeOptimizer();
    void updateWeightsSGD();
    void updateWeightsRMSProp();
    void updateWeightsAdam();
};


class Layer {
protected:
    vector<Neuron*> neurons;
    ActivationFunction activation;
    bool is_output_layer = false;
    bool softmax_enabled = false;

public:
    Layer(int num_neurons, ActivationFunction act, bool has_bias = true);
    ~Layer();
    
    void connectTo(Layer* next_layer);
    void computeOutputs();
    void computeDeltas(const vector<float>* targets = nullptr);
    void updateWeights();
    void applySoftmax();
    
    vector<float> getOutputs() const;
    void setInputs(const vector<float>& inputs);
    void setAsOutputLayer(bool softmax = false);
    
    size_t size() const { return neurons.size(); }
    Neuron* operator[](size_t index) { return neurons[index]; }
    const Neuron* operator[](size_t index) const { return neurons[index]; }

    // Nuevas funciones para guardar/cargar pesos
    void saveWeights(ofstream& file) const;
    void loadWeights(ifstream& file);
};

class MultilayerPerceptron {
    vector<Layer*> layers;
    bool softmax_output = false;
    OptimizerType optimizer_type = OptimizerType::SGD;

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
                                     const vector<vector<float>>& test_inputs,
                                     const vector<vector<float>>& test_targets,
                                     int epochs, int batch_size,
                                     const string& metrics_filename) ;
    float calculateLoss(const vector<float>& output, const vector<float>& target) const;
    float calculateAccuracy(const vector<vector<float>>& inputs, 
                          const vector<vector<float>>& targets) const;
    
    // Funciones para guardar/cargar modelo
    void saveModel(const string& filename) const;
    void loadModel(const string& filename);
    
    void printNetwork() const;



    // Nueva función para configurar el optimizador
    void setOptimizer(OptimizerType type, float learning_rate = LEARNING_RATE) {
        optimizer_type = type;
        for (auto& layer : layers) {
            for (size_t i = 0; i < layer->size(); ++i) {
                Neuron* neuron = (*layer)[i];
                neuron->optimizer = type;
                neuron->learning_rate = learning_rate;
                neuron->initializeOptimizer();
            }
        }
    }


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

// Clase para manejar datos MNIST
class MNISTDataset {
public:
    static vector<vector<float>> loadImages(const string& filename, int max_images = -1);
    static vector<vector<float>> loadLabels(const string& filename, int max_labels = -1);
    static void displayImage(const vector<float>& image, int rows = 28, int cols = 28);
};

// ==================== IMPLEMENTACIONES ====================

// -------------------- Funciones de Activación --------------------
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
        return ActivationFunction(nullptr, nullptr); // Caso especial
    }
}

// -------------------- Neurona --------------------
Neuron::Neuron(ActivationFunction act) : activation(act) {}

void Neuron::initializeOptimizer() {
    if (is_bias) return;
    
    size_t num_weights = inputs.size();
    
    switch (optimizer) {
        case OptimizerType::RMSProp:
            squared_gradients.resize(num_weights, 0.0f);
            break;
        case OptimizerType::Adam:
            m.resize(num_weights, 0.0f);
            v.resize(num_weights, 0.0f);
            t = 0;
            break;
        default:
            // SGD no necesita inicialización
            break;
    }
}


void Neuron::computeOutput() {
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
}

void Neuron::computeDelta(bool is_output_neuron, float target) {
    if (is_output_neuron) {
        delta = (output - target); // Para softmax, la derivada ya está considerada
    } else {
        float sum = 0.0f;
        for (const auto& [neuron, weight] : outputs) {
            sum += *weight * neuron->delta;
        }
        delta = activation.derive(output) * sum;
    }
}

void Neuron::updateWeights() {
    if (is_bias) return;
    
    switch (optimizer) {
        case OptimizerType::SGD:
            updateWeightsSGD();
            break;
        case OptimizerType::RMSProp:
            updateWeightsRMSProp();
            break;
        case OptimizerType::Adam:
            updateWeightsAdam();
            break;
    }
}

void Neuron::updateWeightsSGD() {
    for (auto& [neuron, weight] : inputs) {
        *weight -= learning_rate * delta * neuron->output;
    }
}

void Neuron::updateWeightsRMSProp() {
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& [neuron, weight] = inputs[i];
        float gradient = delta * neuron->output;
        
        // Actualizar la media móvil de los gradientes al cuadrado
        squared_gradients[i] = rmsprop_rho * squared_gradients[i] + 
                              (1 - rmsprop_rho) * gradient * gradient;
        
        // Actualizar el peso
        *weight -= learning_rate * gradient / (sqrt(squared_gradients[i]) + rmsprop_epsilon);
    }
}

void Neuron::updateWeightsAdam() {
    t++;  // Incrementar el contador de pasos
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& [neuron, weight] = inputs[i];
        float gradient = delta * neuron->output;
        
        // Actualizar momentos
        m[i] = adam_beta1 * m[i] + (1 - adam_beta1) * gradient;
        v[i] = adam_beta2 * v[i] + (1 - adam_beta2) * gradient * gradient;
        
        // Calcular momentos corregidos de sesgo
        float m_hat = m[i] / (1 - pow(adam_beta1, t));
        float v_hat = v[i] / (1 - pow(adam_beta2, t));
        
        // Actualizar el peso
        *weight -= learning_rate * m_hat / (sqrt(v_hat) + adam_epsilon);
    }
}

// -------------------- Capa --------------------
Layer::Layer(int num_neurons, ActivationFunction act, bool has_bias) : activation(act) {
    for (int i = 0; i < num_neurons; ++i) {
        neurons.push_back(new Neuron(act));
    }
    if (has_bias) {
        Neuron* bias = new Neuron(act);
        bias->is_bias = true;
        bias->output = 1.0f;
        neurons.push_back(bias);
    }
}

Layer::~Layer() {
    for (auto& neuron : neurons) {
        for (auto& [_, weight] : neuron->inputs) {
            delete weight;
        }
        delete neuron;
    }
}

void Layer::connectTo(Layer* next_layer) {
    srand(RANDOM_SEED);
    for (auto& neuron : next_layer->neurons) {
        if (neuron->is_bias) continue;
        for (auto& prev_neuron : neurons) {
            float* weight = new float((rand()%100)/100.0f - 0.5f);
            neuron->inputs.emplace_back(prev_neuron, weight);
            prev_neuron->outputs.emplace_back(neuron, weight);
        }
    }
}

void Layer::computeOutputs() {
    for (auto& neuron : neurons) {
        neuron->computeOutput();
    }
}

void Layer::computeDeltas(const vector<float>* targets) {
    if (is_output_layer) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            if (!neurons[i]->is_bias) {
                neurons[i]->computeDelta(true, targets ? (*targets)[i] : 0.0f);
            }
        }
    } else {
        for (auto& neuron : neurons) {
            if (!neuron->is_bias) {
                neuron->computeDelta(false);
            }
        }
    }
}

void Layer::updateWeights() {
    for (auto& neuron : neurons) {
        neuron->updateWeights();
    }
}

void Layer::applySoftmax() {
    if (!softmax_enabled) return;
    
    vector<float> z;
    for (auto& neuron : neurons) {
        if (!neuron->is_bias) {
            z.push_back(neuron->net_input);
        }
    }

    auto softmax_values = Activations::softmax(z);
    
    size_t idx = 0;
    for (auto& neuron : neurons) {
        if (!neuron->is_bias) {
            neuron->output = softmax_values[idx++];
        }
    }
}

vector<float> Layer::getOutputs() const {
    vector<float> output;
    for (auto& neuron : neurons) {
        if (!neuron->is_bias) {
            output.push_back(neuron->output);
        }
    }
    return output;
}

void Layer::setInputs(const vector<float>& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        neurons[i]->output = inputs[i];
    }
}

void Layer::setAsOutputLayer(bool softmax) {
    is_output_layer = true;
    softmax_enabled = softmax;
}

void Layer::saveWeights(ofstream& file) const {
    for (auto& neuron : neurons) {
        if (neuron->is_bias) continue;
        for (auto& [_, weight] : neuron->inputs) {
            file.write(reinterpret_cast<const char*>(weight), sizeof(float));
        }
    }
}

void Layer::loadWeights(ifstream& file) {
    for (auto& neuron : neurons) {
        if (neuron->is_bias) continue;
        for (auto& [_, weight] : neuron->inputs) {
            file.read(reinterpret_cast<char*>(weight), sizeof(float));
        }
    }
}

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

    // Configurar el optimizador después de crear las capas
    setOptimizer(optimizer_type, LEARNING_RATE);
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
                                     const vector<vector<float>>& test_inputs,
                                     const vector<vector<float>>& test_targets,
                                     int epochs, int batch_size,
                                     const string& metrics_filename) {
    if (inputs.size() != targets.size()) {
        throw runtime_error("Inputs and targets must have the same size");
    }

    if (!test_inputs.empty() && test_inputs.size() != test_targets.size()) {
        throw runtime_error("Test inputs and test targets must have the same size");
    }
    // Abrir archivo para guardar las métricas
    ofstream metrics_file(metrics_filename);
    if (!metrics_file) {
        cerr << "Warning: Could not open metrics file for writing. Metrics will not be saved." << endl;
    } else {
        // Escribir cabecera del archivo
        metrics_file << "Epoch\tTrain Loss\tTrain Accuracy(%)\tTest Loss\tTest Accuracy(%)\n";
    }

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

        // Calcular métricas
        float avg_loss = total_loss / inputs.size();
        float accuracy = static_cast<float>(correct) / inputs.size() * 100.0f;


         // Evaluación en test
        float test_loss = 0.0f;
        int test_correct = 0;
        
        if (!test_inputs.empty()) {
            #pragma omp parallel for reduction(+:test_loss, test_correct)
            for (size_t i = 0; i < test_inputs.size(); ++i) {
                setInput(test_inputs[i]);
                auto output = forwardPropagate();
                
                test_loss += calculateLoss(output, test_targets[i]);
                
                int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
                int actual = distance(test_targets[i].begin(), max_element(test_targets[i].begin(), test_targets[i].end()));
                if (predicted == actual) test_correct++;
            }
        }

        float avg_test_loss = test_inputs.empty() ? 0.0f : test_loss / test_inputs.size();
        float test_accuracy = test_inputs.empty() ? 0.0f : static_cast<float>(test_correct) / test_inputs.size() * 100.0f;



        // Mostrar estadísticas de la época
        cout << "Epoch " << epoch + 1 << "/" << epochs 
             << " - TrainLoss: " << avg_loss 
             << " - TrainAccuracy: " << accuracy << "%"
             << " - TestLoss: " << avg_test_loss 
             << " - TestAccuracy: " << test_accuracy << "%" << endl;

        // Guardar métricas en el archivo
        if (metrics_file) {
            metrics_file << epoch + 1 << "\t" << avg_loss << "\t" << accuracy << "\t" << avg_test_loss << "\t" << test_accuracy << "\n";
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

    // Guardar arquitectura
    vector<int> architecture;
    for (size_t i = 0; i < layers.size(); ++i) {
        int neurons = layers[i]->size();
        if (i != layers.size()-1) neurons--; // Excluir bias
        architecture.push_back(neurons);
    }
    
    size_t num_layers = architecture.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(architecture.data()), num_layers * sizeof(int));

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

    // Recrear la red
    vector<ActivationFunction> activations(num_layers - 1, Activations::Sigmoid());
    if (num_layers > 1) activations.back() = Activations::Softmax();
    
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

// -------------------- MNIST Dataset --------------------
vector<vector<float>> MNISTDataset::loadImages(const string& filename, int max_images) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open MNIST images file: " + filename);

    // Leer cabecera
    int32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    // Convertir de big-endian a little-endian
    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }

    vector<vector<float>> images;
    images.reserve(num_images);

    for (int i = 0; i < num_images; ++i) {
        vector<float> image(rows * cols);
        for (int j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = pixel / 255.0f; // Normalizar a [0,1]
        }
        images.push_back(move(image));
    }

    return images;
}

vector<vector<float>> MNISTDataset::loadLabels(const string& filename, int max_labels) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open MNIST labels file: " + filename);

    // Leer cabecera
    int32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    // Convertir de big-endian a little-endian
    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    if (max_labels > 0 && max_labels < num_labels) {
        num_labels = max_labels;
    }

    vector<vector<float>> labels;
    labels.reserve(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        
        vector<float> one_hot(10, 0.0f);
        one_hot[label] = 1.0f;
        labels.push_back(move(one_hot));
    }

    return labels;
}

void MNISTDataset::displayImage(const vector<float>& image, int rows, int cols) {
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

// ==================== EJEMPLOS DE USO ====================
int main() {
    // Elegir dataset: MNIST o Fashion-MNIST
    const string dataset_path = "../dataset/mnist/"; // Cambia a "../dataset/fashion-mnist/" si quieres
    const string prefix = "mnist"; // Cambia a "fashion" si es Fashion-MNIST

    const int full_epochs = 20  ;
    const int half_epochs = full_epochs / 2;
    const int double_epochs = full_epochs * 2;
    const int training_samples = 60000;
    const int test_samples = 10000;

    // Variables de tiempo
    chrono::time_point<chrono::system_clock> start, end;
    chrono::duration<double> total_time, load_time, train_time, test_time;

    start = chrono::system_clock::now();

    try {
        cout << "\nCargando " << prefix << "...\n";
        auto load_start = chrono::system_clock::now();

        auto train_images = MNISTDataset::loadImages(dataset_path + "train-images.idx3-ubyte", training_samples);
        auto train_labels = MNISTDataset::loadLabels(dataset_path + "train-labels.idx1-ubyte", training_samples);
        auto test_images = MNISTDataset::loadImages(dataset_path + "t10k-images.idx3-ubyte", test_samples);
        auto test_labels = MNISTDataset::loadLabels(dataset_path + "t10k-labels.idx1-ubyte", test_samples);

        auto load_end = chrono::system_clock::now();


        load_time = load_end - load_start;


        cout << "Datos cargados:\n";
        cout << " - Imágenes de entrenamiento: " << train_images.size() << "\n";
        cout << " - Etiquetas de entrenamiento: " << train_labels.size() << "\n";
        cout << " - Imágenes de test: " << test_images.size() << "\n";
        cout << " - Etiquetas de test: " << test_labels.size() << "\n";

        cout << "Datos cargados: " << train_images.size() << " ejemplos.\n";

        // Crear red neuronal
        MultilayerPerceptron mlp;
        mlp.createNetwork({784, 64, 32, 10}, {
            Activations::Sigmoid(),
            Activations::Sigmoid(),
            Activations::Softmax()
        });
       mlp.setOptimizer(OptimizerType::Adam, 0.001f);  // Tasa de aprendizaje típica para Adam

        // ===== Entrenamiento completo =====
        cout << "\nEntrenamiento normal (" << full_epochs << " épocas) ADAM...\n";
        auto train_start = chrono::system_clock::now();
        mlp.trainDataset(train_images, train_labels, test_images, test_labels, full_epochs,1, prefix + "_train_" + to_string(full_epochs) +"epochs_adam.csv");
        auto train_end = chrono::system_clock::now();
        train_time = train_end - train_start;

        cout << "Tiempo total entrenamiento: " << train_time.count() << " segundos\n";

        // Evaluación
        auto test_start = chrono::system_clock::now();
        mlp.testModel(test_images, test_labels, false);
        auto test_end = chrono::system_clock::now();
        test_time = test_end - test_start;

        // Guardar modelo
        mlp.saveModel(prefix + "_model_" + to_string(full_epochs) + "epochs.bin");





/*

        /////////////////////////
        // Crear red neuronal
        MultilayerPerceptron mlp2;
        mlp2.createNetwork({784, 64, 32, 10}, {
            Activations::Sigmoid(),
            Activations::Sigmoid(),
            Activations::Softmax()
        });
       mlp2.setOptimizer(OptimizerType::RMSProp, 0.001f);  // Tasa de aprendizaje típica para Adam

        // ===== Entrenamiento completo =====
        cout << "\nEntrenamiento normal (" << full_epochs << " épocas) RMSProp...\n";
        auto train_start2 = chrono::system_clock::now();
        mlp2.trainDataset(train_images, train_labels, test_images, test_labels,  full_epochs,1, prefix + "_train_" + to_string(full_epochs) +"epochs_RMS.csv");
        auto train_end2 = chrono::system_clock::now();
        train_time = train_end2 - train_start2;

        cout << "Tiempo total entrenamiento: " << train_time.count() << " segundos\n";

        // Evaluación
        auto test_start2 = chrono::system_clock::now();
        mlp2.testModel(test_images, test_labels, false);
        auto test_end2 = chrono::system_clock::now();
        test_time = test_end2 - test_start2;

        
        
        
        
        
        
        
        //////////////////////
        // Crear red neuronal
        MultilayerPerceptron mlp3;
        mlp3.createNetwork({784, 64, 32, 10}, {
            Activations::Sigmoid(),
            Activations::Sigmoid(),
            Activations::Softmax()
        });
       mlp3.setOptimizer(OptimizerType::SGD, 0.001f);  // Tasa de aprendizaje típica para Adam

        // ===== Entrenamiento completo =====
        cout << "\nEntrenamiento normal (" << full_epochs << " épocas) SGD ...\n";
        auto train_start3 = chrono::system_clock::now();
        mlp3.trainDataset(train_images, train_labels, test_images, test_labels,  full_epochs,1, prefix + "_train_" + to_string(full_epochs) +"epochs_sgd.csv");
        auto train_end3 = chrono::system_clock::now();
        train_time = train_end3 - train_start3;

        cout << "Tiempo total entrenamiento: " << train_time.count() << " segundos\n";

        // Evaluación
        auto test_start3 = chrono::system_clock::now();
        mlp3.testModel(test_images, test_labels, false);
        auto test_end3 = chrono::system_clock::now();
        test_time = test_end3 - test_start3;

*/

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    end = chrono::system_clock::now();
    total_time = end - start;
    cout << "\nTotal ejecución: " << total_time.count() << " segundos\n";








    auto test_images2 = MNISTDataset::loadImages("train-images.idx3-ubyte", test_samples);
    auto test_labels2 = MNISTDataset::loadLabels("train-labels.idx1-ubyte", test_samples);


    // Crear red neuronal
    MultilayerPerceptron mlp;
    
    // ===== Entrenamiento completo =====
    cout << "\nEntrenamiento normal (" << full_epochs << " épocas)...\n";
    
    // Guardar modelo
    mlp.loadModel("mnist_model_30epochs.bin");
    mlp.testModel(test_images2, test_labels2,false);


    return 0;
}
