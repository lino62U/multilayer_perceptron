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
#include <omp.h>
#include <random>  // Para random_device y mt19937

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;
namespace fs = std::filesystem;

enum class OptimizerType {
    SGD,    // Descenso de gradiente estocástico (por defecto)
    RMSProp,
    Adam
};

// ==================== DECLARACIONES OPTIMIZADAS ====================
using ActivationFunc = float(*)(float);

struct ActivationFunction {
    ActivationFunc activate;
    ActivationFunc derive;

    ActivationFunction(ActivationFunc a = nullptr, ActivationFunc d = nullptr)
        : activate(a), derive(d) {}
};

namespace Activations {
    // Funciones inline para mejor performance
    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline float sigmoid_derivative(float y) {
        return y * (1.0f - y);
    }

    inline float relu(float x) {
        return x > 0 ? x : 0.0f;
    }

    inline float relu_derivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    inline float tanh_act(float x) {
        return std::tanh(x);
    }

    inline float tanh_derivative(float y) {
        return 1.0f - y * y;
    }

    // Optimización: evitar copias y usar move semantics
    inline vector<float> softmax(const vector<float>& z) {
        vector<float> res;
        res.reserve(z.size());
        float max_z = *max_element(z.begin(), z.end());
        float sum = 0.0f;

        for (float val : z) {
            float exp_val = exp(val - max_z); // Resta el máximo para estabilidad numérica
            res.push_back(exp_val);
            sum += exp_val;
        }

        // Asegúrate de que sum no sea cero
        if (sum == 0.0f) sum = 1e-8f;
        
        for (float& val : res) {
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
        return ActivationFunction(nullptr, nullptr);
    }
}

struct Neuron {
    float output = 0.0f;
    float net_input = 0.0f;
    float delta = 0.0f;
    bool is_bias = false;
    vector<pair<Neuron*, float*>> inputs;
    vector<pair<Neuron*, float*>> outputs;
    ActivationFunction activation;
    
    OptimizerType optimizer = OptimizerType::SGD;
    float learning_rate = LEARNING_RATE;
    float weight_decay = 0.0f;
    
    // Para RMSProp
    vector<float> squared_gradients;
    float rmsprop_rho = 0.9f;
    float rmsprop_epsilon = 1e-8f;
    
    // Para Adam
    vector<float> m;
    vector<float> v;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1e-8f;
    int t = 0;

    Neuron(ActivationFunction act = Activations::Sigmoid()) : activation(act) {}

    inline void computeOutput() {
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

    inline void computeDelta(bool is_output_neuron, float target = 0.0f) {
        if (is_output_neuron) {
            delta = (output - target);
        } else {
            float sum = 0.0f;
            for (const auto& [neuron, weight] : outputs) {
                sum += *weight * neuron->delta;
            }
            delta = activation.derive(output) * sum;
        }
    }

    void updateWeights() {
        if (is_bias) return;
        
        switch (optimizer) {
            case OptimizerType::SGD: updateWeightsSGD(); break;
            case OptimizerType::RMSProp: updateWeightsRMSProp(); break;
            case OptimizerType::Adam: updateWeightsAdam(); break;
        }
    }

    void initializeOptimizer() {
        if (is_bias) return;
        
        const size_t num_weights = inputs.size();
        
        switch (optimizer) {
            case OptimizerType::RMSProp:
                squared_gradients.assign(num_weights, 0.0f);
                break;
            case OptimizerType::Adam:
                m.assign(num_weights, 0.0f);
                v.assign(num_weights, 0.0f);
                t = 0;
                break;
            default: break;
        }
    }

private:
    inline void updateWeightsSGD() {
        for (auto& [neuron, weight] : inputs) {
            float gradient = delta * neuron->output;
            if (weight_decay > 0.0f) {
                gradient += weight_decay * (*weight);
            }
            *weight -= learning_rate * gradient;
        }
    }

    inline void updateWeightsRMSProp() {
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& [neuron, weight] = inputs[i];
            float gradient = delta * neuron->output;
            
            if (weight_decay > 0.0f) {
                gradient += weight_decay * (*weight);
            }
            
            squared_gradients[i] = rmsprop_rho * squared_gradients[i] + 
                                 (1 - rmsprop_rho) * gradient * gradient;
            
            *weight -= learning_rate * gradient / (sqrt(squared_gradients[i]) + rmsprop_epsilon);
        }
    }

    inline void updateWeightsAdam() {
        t++;
        const float lr = learning_rate;
        const float beta1_t = 1.0f - powf(adam_beta1, t);
        const float beta2_t = 1.0f - powf(adam_beta2, t);
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& [neuron, weight] = inputs[i];
            
            float grad = delta * neuron->output;
            if (weight_decay > 0.0f) grad += weight_decay * (*weight);
            
            m[i] = adam_beta1 * m[i] + (1.0f - adam_beta1) * grad;
            v[i] = adam_beta2 * v[i] + (1.0f - adam_beta2) * grad * grad;
            
            float m_hat = m[i] / beta1_t;
            float v_hat = v[i] / beta2_t;
            
            *weight -= lr * m_hat / (sqrtf(v_hat) + adam_epsilon);
            
            // Limitar pesos
            *weight = max(-5.0f, min(5.0f, *weight));
        }
    }
};

class Layer {
protected:
    vector<Neuron*> neurons;
    ActivationFunction activation;
    bool is_output_layer = false;
    bool softmax_enabled = false;
    float dropout_rate = 0.0f;
    vector<bool> dropout_mask;
    bool is_training = true;

public:
    Layer(int num_neurons, ActivationFunction act, bool has_bias = true) : activation(act) {
        neurons.reserve(num_neurons + (has_bias ? 1 : 0));
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

    ~Layer() {
        for (auto& neuron : neurons) {
            for (auto& [_, weight] : neuron->inputs) {
                delete weight;
            }
            delete neuron;
        }
    }

    void connectTo(Layer* next_layer) {
        srand(RANDOM_SEED);
        for (auto& neuron : next_layer->neurons) {
            if (neuron->is_bias) continue;
            for (auto& prev_neuron : neurons) {
                //float* weight = new float((rand()%100)/100.0f - 0.5f);
                // Por una inicialización más adecuada (He initialization para ReLU, Xavier para sigmoid):
                float scale = sqrt(2.0f / prev_neuron->outputs.size());
                float* weight = new float((rand()%100)/100.0f * scale - scale/2.0f);
                
                neuron->inputs.emplace_back(prev_neuron, weight);
                prev_neuron->outputs.emplace_back(neuron, weight);
            }
        }
    }

    void computeOutputs() {
        for (auto& neuron : neurons) {
            neuron->computeOutput();
        }
        
        if (dropout_rate > 0.0f && is_training) {
            applyDropout();
        }
    }

    void applyDropout() {
        for (size_t i = 0; i < neurons.size(); ++i) {
            if (!neurons[i]->is_bias) {
                dropout_mask[i] = (static_cast<float>(rand())) / RAND_MAX < dropout_rate;
                neurons[i]->output = dropout_mask[i] ? 0.0f : neurons[i]->output / (1.0f - dropout_rate);
            }
        }
    }

    void computeDeltas(const vector<float>* targets = nullptr) {
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

    void updateWeights() {
        for (auto& neuron : neurons) {
            neuron->updateWeights();
        }
    }

    void applySoftmax() {
        if (!softmax_enabled) return;
        
        vector<float> z;
        z.reserve(neurons.size());
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

    vector<float> getOutputs() const {
        vector<float> output;
        output.reserve(neurons.size());
        for (auto& neuron : neurons) {
            if (!neuron->is_bias) {
                output.push_back(neuron->output);
            }
        }
        return output;
    }

    void setInputs(const vector<float>& inputs) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            neurons[i]->output = inputs[i];
        }
    }

    void setAsOutputLayer(bool softmax = false) {
        is_output_layer = true;
        softmax_enabled = softmax;
    }

    size_t size() const { return neurons.size(); }
    Neuron* operator[](size_t index) { return neurons[index]; }
    const Neuron* operator[](size_t index) const { return neurons[index]; }

    void saveWeights(ofstream& file) const {
        for (auto& neuron : neurons) {
            if (neuron->is_bias) continue;
            for (auto& [_, weight] : neuron->inputs) {
                file.write(reinterpret_cast<const char*>(weight), sizeof(float));
            }
        }
    }

    void loadWeights(ifstream& file) {
        for (auto& neuron : neurons) {
            if (neuron->is_bias) continue;
            for (auto& [_, weight] : neuron->inputs) {
                file.read(reinterpret_cast<char*>(weight), sizeof(float));
            }
        }
    }

    void setDropoutRate(float rate) { 
        dropout_rate = rate; 
        if (rate > 0.0f) {
            dropout_mask.resize(neurons.size(), false);
        }
    }
    
    void setTrainingMode(bool training) { is_training = training; }
};

class MultilayerPerceptron {
    vector<Layer*> layers;
    bool softmax_output = false;
    OptimizerType optimizer_type = OptimizerType::SGD;

public:
    ~MultilayerPerceptron() {
        for (auto& layer : layers) {
            delete layer;
        }
    }

    void createNetwork(const vector<int>& architecture, 
                      const vector<ActivationFunction>& activations) {
        if (activations.size() != architecture.size() - 1) {
            throw runtime_error("Number of activation functions must match hidden layers + output");
        }
        
        softmax_output = (activations.back().activate == nullptr);

        // Crear capas
        layers.reserve(architecture.size());
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

        setOptimizer(optimizer_type, LEARNING_RATE);
    }

    void setInput(const vector<float>& inputs) const {
        layers[0]->setInputs(inputs);
    }

    vector<float> forwardPropagate() const {
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i]->computeOutputs();
            if (i == layers.size()-1 && softmax_output) {
                layers[i]->applySoftmax();
            }
        }
        return layers.back()->getOutputs();
    }

    void backPropagate(const vector<float>& targets) {
        layers.back()->computeDeltas(&targets);

        for (int i = layers.size()-2; i >= 0; --i) {
            layers[i]->computeDeltas();
        }

        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i]->updateWeights();
        }
    }

    void train(const vector<float>& input, const vector<float>& target) {
        setInput(input);
        forwardPropagate();
        backPropagate(target);
    }

    void trainDataset(const vector<vector<float>>& inputs, 
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

        ofstream metrics_file(metrics_filename);
        if (metrics_file) {
            metrics_file << "Epoch\tTrain Loss\tTrain Accuracy(%)\tTest Loss\tTest Accuracy(%)\n";
        }

        const size_t num_samples = inputs.size();
        vector<size_t> indices(num_samples);
        iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Mezclar índices para el epoch
            shuffle(indices.begin(), indices.end(), mt19937(random_device()()));
            
            float total_loss = 0.0f;
            int correct = 0;

            setTrainingMode(true);

            // Entrenamiento por lotes
            for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
                size_t batch_end = min(batch_start + batch_size, num_samples);
                
                // Procesar batch en paralelo
                #pragma omp parallel for reduction(+:total_loss, correct)
                for (size_t i = batch_start; i < batch_end; ++i) {
                    size_t idx = indices[i];
                    setInput(inputs[idx]);
                    auto output = forwardPropagate();
                    backPropagate(targets[idx]);

                    total_loss += calculateLoss(output, targets[idx]);
                    
                    int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
                    int actual = distance(targets[idx].begin(), max_element(targets[idx].begin(), targets[idx].end()));
                    if (predicted == actual) correct++;
                }
            }

            float avg_loss = total_loss / num_samples;
            float accuracy = static_cast<float>(correct) / num_samples * 100.0f;

            // Evaluación en test
            float test_loss = 0.0f;
            int test_correct = 0;
            
            if (!test_inputs.empty()) {
                setTrainingMode(false);
                
                #pragma omp parallel for reduction(+:test_loss, test_correct)
                for (size_t i = 0; i < test_inputs.size(); ++i) {
                    setInput(test_inputs[i]);
                    auto output = forwardPropagate();
                    
                    test_loss += calculateLoss(output, test_targets[i]);
                    
                    int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
                    int actual = distance(test_targets[i].begin(), max_element(test_targets[i].begin(), test_targets[i].end()));
                    if (predicted == actual) test_correct++;
                }

                setTrainingMode(true);
            }

            float avg_test_loss = test_inputs.empty() ? 0.0f : test_loss / test_inputs.size();
            float test_accuracy = test_inputs.empty() ? 0.0f : static_cast<float>(test_correct) / test_inputs.size() * 100.0f;

            cout << "Epoch " << epoch + 1 << "/" << epochs 
                 << " - Loss: " << avg_loss 
                 << " - Acc: " << accuracy << "%"
                 << " - Val Loss: " << avg_test_loss 
                 << " - Val Acc: " << test_accuracy << "%" << endl;

            if (metrics_file) {
                metrics_file << epoch + 1 << "\t" << avg_loss << "\t" << accuracy << "\t" << avg_test_loss << "\t" << test_accuracy << "\n";
            }
        }

        setTrainingMode(false);
    }

    inline float calculateLoss(const vector<float>& output, const vector<float>& target) const {
        // Usar cross-entropy loss para softmax
        float loss = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            loss -= target[i] * log(max(output[i], 1e-8f)); // Evitar log(0)
        }
        return loss;
    }

    float calculateAccuracy(const vector<vector<float>>& inputs, 
                         const vector<vector<float>>& targets) const {
        if (inputs.size() != targets.size()) {
            throw runtime_error("Inputs and targets must have the same size");
        }

        int correct = 0;
        #pragma omp parallel for reduction(+:correct)
        for (size_t i = 0; i < inputs.size(); ++i) {
            setInput(inputs[i]);
            auto output = forwardPropagate();
            
            int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
            //int actual = distance(targets[i].begin(), max_element(targets[i].begin(), targets[i].end()));
            int actual = 0;
            for (size_t k = 0; k < targets[i].size(); ++k) {
                if (targets[i][k] == 1.0f) {
                    actual = k;
                    break;
                }
            }
            
            if (predicted == actual) correct++;
        }

        return static_cast<float>(correct) / inputs.size() * 100.0f;
    }

    void saveModel(const string& filename) const {
        ofstream file(filename, ios::binary);
        if (!file) {
            throw runtime_error("Cannot open file for writing: " + filename);
        }

        // Guardar arquitectura
        vector<int> architecture;
        architecture.reserve(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            int neurons = layers[i]->size();
            if (i != layers.size()-1) neurons--;
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

    void loadModel(const string& filename) {
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
    
    void setWeightDecay(float decay) {
        for (auto& layer : layers) {
            for (size_t i = 0; i < layer->size(); ++i) {
                Neuron* neuron = (*layer)[i];
                neuron->weight_decay = decay;
            }
        }
    }
    
    void setDropoutRate(float rate) {
        for (size_t i = 1; i < layers.size() - 1; ++i) {
            layers[i]->setDropoutRate(rate);
        }
    }
    
    void setTrainingMode(bool training) {
        for (auto& layer : layers) {
            layer->setTrainingMode(training);
        }
    }

    void testModel(const vector<vector<float>>& test_images, 
                  const vector<vector<float>>& test_labels,
                  bool show_details = false) const {
        if (test_images.size() != test_labels.size()) {
            throw runtime_error("Test images and labels must have the same size");
        }

        int correct = 0;
        #pragma omp parallel for reduction(+:correct)
        for (size_t i = 0; i < test_images.size(); ++i) {
            setInput(test_images[i]);
            auto output = forwardPropagate();

            int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
            //int actual = distance(test_labels[i].begin(), max_element(test_labels[i].begin(), test_labels[i].end()));
            int actual = 0;
            for (size_t k = 0; k < test_labels[i].size(); ++k) {
                if (test_labels[i][k] == 1.0f) {
                    actual = k;
                    break;
                }
            }

            if (show_details) {
                #pragma omp critical
                {
                    cout << "Test Sample #" << i + 1 << "\n";
                    displayImage(test_images[i]);
                    cout << "Actual: " << actual << " | Predicted: " << predicted << "\n\n";
                }
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
    const int training_samples = 10000;
    const int test_samples = 5000;

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
        mlp.createNetwork({784, 128, 64, 10}, {
            Activations::Sigmoid(),
            Activations::Sigmoid(),
            Activations::Softmax()
        });
       mlp.setOptimizer(OptimizerType::Adam, 0.001f);  // Tasa de aprendizaje típica para Adam
       //mlp.setWeightDecay(0.001f);  // Configurar weight decay
       //mlp.setDropoutRate(0.2f);  // 50% de dropout en capas ocultas

       //mlp.setTrainingMode(true);  // Activar dropout

        // ===== Entrenamiento completo =====
        cout << "\nEntrenamiento normal (" << full_epochs << " épocas) ADAM...\n";
        auto train_start = chrono::system_clock::now();
        mlp.trainDataset(train_images, train_labels, test_images, test_labels, full_epochs,1, prefix + "_train_" + to_string(full_epochs) +"epochs_adam_decay.csv");
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






    return 0;
}
