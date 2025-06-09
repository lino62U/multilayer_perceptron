#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
//#include <omp.h>
#include <cassert>  // Added for assert
#include <cstdint>  // Added for uint8_t
#include <random>
#include <mutex>
#include <memory>
#include <thread>
using namespace std;

// ==================== CONSTANTS ====================
constexpr float LEARNING_RATE = 0.1f;
constexpr int RANDOM_SEED = 42;
constexpr int DEFAULT_BATCH_SIZE = 1;

// ==================== TYPE ALIASES ====================
using Matrix = vector<vector<float>>;
using Vector = vector<float>;

// ==================== ACTIVATION FUNCTIONS ====================
struct Activation {
    function<float(float)> activate;
    function<float(float)> derive;
    
    Activation(function<float(float)> a, function<float(float)> d) 
        : activate(a), derive(d) {}
};

namespace Activations {
    // Funciones de activación
    static float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
    static float sigmoid_derivative(float y) { return y * (1.0f - y); }
    static float relu(float x) { return x > 0 ? x : 0.0f; }
    static float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }
    static float tanh_act(float x) { return std::tanh(x); }
    static float tanh_derivative(float y) { return 1.0f - y * y; }
    
    static Vector softmax(const Vector& z) {
        Vector res(z.size());
        float max_z = *max_element(z.begin(), z.end());
        float sum = 0.0f;

        for (size_t i = 0; i < z.size(); ++i) {
            res[i] = exp(z[i] - max_z);
            sum += res[i];
        }

        for (auto& val : res) val /= sum;
        return res;
    }
    
    static Activation Sigmoid() { return Activation(sigmoid, sigmoid_derivative); }
    static Activation ReLU() { return Activation(relu, relu_derivative); }
    static Activation Tanh() { return Activation(tanh_act, tanh_derivative); }
    // Función de construcción para Softmax
    static Activation Softmax() {
        return Activation(nullptr, nullptr);
    }
}

// ==================== OPTIMIZERS ====================

class Optimizer {
protected:
    float learning_rate;
    float beta1;  // Para Adam
    float beta2;  // Para Adam y RMSProp
    float epsilon;
    int t;  // Paso de tiempo
    
public:
    Optimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}
    
    virtual ~Optimizer() = default;
    
    virtual void updateWeights(Matrix& weights, const Matrix& weight_grads, 
                             Vector& biases, const Vector& bias_grads) = 0;
    
    virtual void initializeStates(size_t rows, size_t cols) = 0;
};

class SGD : public Optimizer {
public:
    SGD(float lr = 0.01f) : Optimizer(lr) {}
    
    void updateWeights(Matrix& weights, const Matrix& weight_grads, 
                      Vector& biases, const Vector& bias_grads) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learning_rate * weight_grads[i][j];
            }
            biases[i] -= learning_rate * bias_grads[i];
        }
    }
    
    void initializeStates(size_t rows, size_t cols) override {
        // SGD no necesita estados adicionales
    }
};

class RMSProp : public Optimizer {
private:
    Matrix s_weights;  // Promedio móvil de los gradientes al cuadrado
    Vector s_biases;
    
public:
    RMSProp(float lr = 0.001f, float b2 = 0.9f, float eps = 1e-8f) 
        : Optimizer(lr, 0.0f, b2, eps) {}
    
    void initializeStates(size_t rows, size_t cols) override {
        s_weights.resize(rows, Vector(cols, 0.0f));
        s_biases.resize(rows, 0.0f);
    }
    
    void updateWeights(Matrix& weights, const Matrix& weight_grads, 
                      Vector& biases, const Vector& bias_grads) override {
        t++;
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                s_weights[i][j] = beta2 * s_weights[i][j] + (1.0f - beta2) * weight_grads[i][j] * weight_grads[i][j];
                weights[i][j] -= learning_rate * weight_grads[i][j] / (sqrt(s_weights[i][j]) + epsilon);
            }
            
            s_biases[i] = beta2 * s_biases[i] + (1.0f - beta2) * bias_grads[i] * bias_grads[i];
            biases[i] -= learning_rate * bias_grads[i] / (sqrt(s_biases[i]) + epsilon);
        }
    }
};

class Adam : public Optimizer {
private:
    Matrix m_weights;  // Promedio móvil de los gradientes
    Matrix v_weights;  // Promedio móvil de los gradientes al cuadrado
    Vector m_biases;
    Vector v_biases;
    
public:
    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : Optimizer(lr, b1, b2, eps) {}
    
    void initializeStates(size_t rows, size_t cols) override {
        m_weights.resize(rows, Vector(cols, 0.0f));
        v_weights.resize(rows, Vector(cols, 0.0f));
        m_biases.resize(rows, 0.0f);
        v_biases.resize(rows, 0.0f);
    }
    
    void updateWeights(Matrix& weights, const Matrix& weight_grads, 
                      Vector& biases, const Vector& bias_grads) override {
        t++;
        float beta1_t = pow(beta1, t);
        float beta2_t = pow(beta2, t);
        
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                m_weights[i][j] = beta1 * m_weights[i][j] + (1.0f - beta1) * weight_grads[i][j];
                v_weights[i][j] = beta2 * v_weights[i][j] + (1.0f - beta2) * weight_grads[i][j] * weight_grads[i][j];
                
                // Corrección de bias
                float m_hat = m_weights[i][j] / (1.0f - beta1_t);
                float v_hat = v_weights[i][j] / (1.0f - beta2_t);
                
                weights[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            
            m_biases[i] = beta1 * m_biases[i] + (1.0f - beta1) * bias_grads[i];
            v_biases[i] = beta2 * v_biases[i] + (1.0f - beta2) * bias_grads[i] * bias_grads[i];
            
            // Corrección de bias
            float m_hat = m_biases[i] / (1.0f - beta1_t);
            float v_hat = v_biases[i] / (1.0f - beta2_t);
            
            biases[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
};

// ==================== LAYER ====================

class Layer {
private:
    Matrix weights;
    Vector biases;
    Vector outputs;
    Vector deltas;
    Vector z;  // net_inputs
    Activation activation;
    bool is_output = false;
    bool use_softmax = false;
    
    // Para optimizadores
    unique_ptr<Optimizer> optimizer;
    Matrix weight_grads;
    Vector bias_grads;
    
public:
    Layer(int input_size, int output_size, Activation act, bool output_layer = false, 
          unique_ptr<Optimizer> opt = make_unique<SGD>(LEARNING_RATE)) 
        : activation(act), is_output(output_layer), optimizer(move(opt)) {
        // Inicialización de pesos y biases
        weights.resize(output_size, vector<float>(input_size));
        biases.resize(output_size);
        
        srand(RANDOM_SEED);
        for (auto& row : weights) {
            for (auto& w : row) {
                w = (rand()%100)/100.0f - 0.5f;
            }
        }
        for (auto& b : biases) {
            b = (rand()%100)/100.0f - 0.5f;
        }
        
        outputs.resize(output_size);
        deltas.resize(output_size);
        z.resize(output_size);
        
        // Inicializar gradientes y estados del optimizador
        weight_grads.resize(output_size, vector<float>(input_size, 0.0f));
        bias_grads.resize(output_size, 0.0f);
        optimizer->initializeStates(output_size, input_size);
    }
    
    void setUseSoftmax(bool use) { use_softmax = use; }
    void setOptimizer(unique_ptr<Optimizer> opt) { 
        optimizer = move(opt);
        optimizer->initializeStates(weights.size(), weights[0].size());
    }
    
    // Forward pass para un batch de inputs
    Matrix forward(const Matrix& inputs) {
        Matrix batch_outputs(inputs.size(), Vector(weights.size()));
        
        for (size_t b = 0; b < inputs.size(); ++b) {
            for (size_t l = 0; l < weights.size(); ++l) {
                z[l] = inner_product(weights[l].begin(), weights[l].end(), 
                                inputs[b].begin(), biases[l]);
                
                outputs[l] = use_softmax ? z[l] : activation.activate(z[l]);
            }
            
            if (use_softmax) {
                Vector softmax_out = Activations::softmax(outputs);
                copy(softmax_out.begin(), softmax_out.end(), 
                    batch_outputs[b].begin());
            } else {
                copy(outputs.begin(), outputs.end(), 
                    batch_outputs[b].begin());
            }
        }
        
        return batch_outputs;
    }
    
    // Backward pass
    void backward(const Matrix& inputs, const Matrix& next_layer_deltas, 
             const Matrix& next_layer_weights) {
        if (next_layer_deltas.empty() || next_layer_weights.empty()) {
            throw runtime_error("Empty matrices in backward pass");
        }

        for (size_t b = 0; b < inputs.size(); ++b) {
            if (b >= next_layer_deltas.size()) break;
            
            for (size_t k = 0; k < deltas.size(); ++k) {
                float sum = 0.0f;
                for (size_t l = 0; l < next_layer_weights.size(); ++l) {
                    if (l < next_layer_deltas[b].size()) {
                        sum += next_layer_deltas[b][l] * next_layer_weights[l][k];
                    }
                }
                deltas[k] = sum * activation.derive(outputs[k]);
            }
        }
    }
    
    // Backward para capa de salida
    void backwardOutput(const Matrix& inputs, const Matrix& targets) {
        for (size_t b = 0; b < inputs.size(); ++b) {
            for (size_t k = 0; k < deltas.size(); ++k) {
                if (use_softmax) {
                    deltas[k] = outputs[k] - targets[b][k];
                } else {
                    deltas[k] = (outputs[k] - targets[b][k]) * activation.derive(outputs[k]);
                }
            }
        }
    }
    
    // Calcular gradientes
    void computeGradients(const Matrix& inputs) {
        // Resetear gradientes
        for (auto& row : weight_grads) fill(row.begin(), row.end(), 0.0f);
        fill(bias_grads.begin(), bias_grads.end(), 0.0f);
        
        // Calcular gradientes para el batch
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t b = 0; b < inputs.size(); ++b) {
                for (size_t k = 0; k < weights[l].size(); ++k) {
                    weight_grads[l][k] += deltas[l] * inputs[b][k];
                }
                bias_grads[l] += deltas[l];
            }
            
            // Promediar los gradientes del batch
            for (size_t k = 0; k < weights[l].size(); ++k) {
                weight_grads[l][k] /= inputs.size();
            }
            bias_grads[l] /= inputs.size();
        }
    }
    
    // Actualizar pesos usando el optimizador
    void updateWeights(const Matrix& inputs) {
        computeGradients(inputs);
        optimizer->updateWeights(weights, weight_grads, biases, bias_grads);
    }
    
    const Vector& getOutputs() const { return outputs; }
    const Vector& getDeltas() const { return deltas; }
    const Matrix& getWeights() const { return weights; }
    const Vector& getBiases() const { return biases; }

    Matrix& getWeights() { return weights; }
    Vector& getBiases() { return biases; }
    Matrix getBatchOutputs() const { return {outputs}; }
};

// ==================== PARALLEL MLP ====================

class ParallelMLP {
private:
    vector<Layer> layers;
    int batch_size;
    int num_threads;
    
public:
    ParallelMLP(const vector<int>& architecture, 
               const vector<Activation>& activations,
               int batch_size = 32, 
               int threads = thread::hardware_concurrency(),
               const string& optimizer_type = "adam",
               float learning_rate = 0.001f)
        : batch_size(batch_size), num_threads(threads) {
        
        for (size_t i = 1; i < architecture.size(); ++i) {
            bool is_output = (i == architecture.size() - 1);
            
            unique_ptr<Optimizer> opt;
            if (optimizer_type == "adam") {
                opt = make_unique<Adam>(learning_rate);
            } else if (optimizer_type == "rmsprop") {
                opt = make_unique<RMSProp>(learning_rate);
            } else {
                opt = make_unique<SGD>(learning_rate);
            }
            
            layers.emplace_back(architecture[i-1], architecture[i], 
                               activations[i-1], is_output, move(opt));
            
            if (is_output && activations[i-1].activate == nullptr) {
                layers.back().setUseSoftmax(true);
            }
        }

        if (architecture.size() < 2 || architecture.size() != activations.size() + 1) {
            throw runtime_error("Invalid architecture or activations size");
        }
    }
    
    void setOptimizer(const string& optimizer_type, float learning_rate) {
        for (auto& layer : layers) {
            unique_ptr<Optimizer> opt;
            if (optimizer_type == "adam") {
                opt = make_unique<Adam>(learning_rate);
            } else if (optimizer_type == "rmsprop") {
                opt = make_unique<RMSProp>(learning_rate);
            } else {
                opt = make_unique<SGD>(learning_rate);
            }
            layer.setOptimizer(move(opt));
        }
    }
    
    // Resto de los métodos permanecen iguales...
    Matrix forward(const Matrix& inputs) {
        if (inputs.empty() || inputs[0].empty()) {
            throw runtime_error("Empty input matrix");
        }
        
        Matrix current_inputs = inputs;
        
        for (auto& layer : layers) {
            current_inputs = layer.forward(current_inputs);
            if (current_inputs.empty() || current_inputs[0].empty()) {
                throw runtime_error("Layer produced empty output");
            }
        }
        
        return current_inputs;
    }
    
    void backward(const Matrix& inputs, const Matrix& targets) {
        forward(inputs);
        
        if (layers.empty()) return;
        
        const Matrix& last_hidden_output = layers.size() > 1 ? 
            Matrix{layers[layers.size()-2].getOutputs()} : inputs;
        
        layers.back().backwardOutput(last_hidden_output, targets);
        
        for (int i = layers.size() - 2; i >= 0; --i) {
            const Matrix& prev_outputs = i > 0 ? 
                Matrix{layers[i-1].getOutputs()} : inputs;
                
            layers[i].backward(
                prev_outputs,
                Matrix{layers[i+1].getDeltas()},
                layers[i+1].getWeights()
            );
        }
        
        for (size_t i = 0; i < layers.size(); ++i) {
            const Matrix& layer_inputs = (i == 0) ? inputs : Matrix{layers[i-1].getOutputs()};
            layers[i].updateWeights(layer_inputs);
        }
    }
    
    void trainParallel(const Matrix& all_inputs, const Matrix& all_targets, 
                      int epochs, const string& metrics_file = "") {
        ofstream metrics;
        if (!metrics_file.empty()) {
            metrics.open(metrics_file);
            metrics << "Epoch,Loss,Accuracy\n";
        }
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int correct = 0;
            
            for (size_t b = 0; b < all_inputs.size(); b += batch_size) {
                size_t batch_end = min(b + batch_size, all_inputs.size());

                if (batch_end > all_inputs.size() || batch_end > all_targets.size()) {
                    throw runtime_error("Batch access out of range");
                }
                Matrix batch_inputs(all_inputs.begin() + b, all_inputs.begin() + batch_end);
                Matrix batch_targets(all_targets.begin() + b, all_targets.begin() + batch_end);
                
                backward(batch_inputs, batch_targets);
                
                auto outputs = forward(batch_inputs);
                for (size_t i = 0; i < outputs.size(); ++i) {
                    for (size_t j = 0; j < outputs[i].size(); ++j) {
                        total_loss += 0.5f * pow(outputs[i][j] - batch_targets[i][j], 2);
                    }
                    
                    int predicted = distance(outputs[i].begin(), 
                                          max_element(outputs[i].begin(), outputs[i].end()));
                    int actual = distance(batch_targets[i].begin(), 
                                       max_element(batch_targets[i].begin(), batch_targets[i].end()));
                    if (predicted == actual) correct++;
                }
            }
            
            float avg_loss = total_loss / all_inputs.size();
            float accuracy = 100.0f * correct / all_inputs.size();
            
            cout << "Epoch " << epoch + 1 << "/" << epochs 
                 << " - Loss: " << avg_loss 
                 << " - Accuracy: " << accuracy << "%" << endl;
                 
            if (metrics.is_open()) {
                metrics << epoch + 1 << "," << avg_loss << "," << accuracy << "\n";
            }
        }
    }
    
    float evaluate(const Matrix& inputs, const Matrix& targets) {
        int correct = 0;
        
        for (size_t b = 0; b < inputs.size(); b += batch_size) {
            size_t batch_end = min(b + batch_size, inputs.size());
            Matrix batch_inputs(inputs.begin() + b, inputs.begin() + batch_end);
            
            auto outputs = forward(batch_inputs);
            
            for (size_t i = 0; i < outputs.size(); ++i) {
                int predicted = distance(outputs[i].begin(), 
                                      max_element(outputs[i].begin(), outputs[i].end()));
                int actual = distance(targets[b + i].begin(), 
                                   max_element(targets[b + i].begin(), targets[b + i].end()));
                if (predicted == actual) correct++;
            }
        }
        
        return 100.0f * correct / inputs.size();
    }
    
    void saveModel(const string& filename) {
        ofstream file(filename, ios::binary);
        if (!file) throw runtime_error("Cannot open file for writing");
        
        vector<int> architecture;
        if (!layers.empty()) {
            architecture.push_back(layers[0].getWeights()[0].size());
            for (const auto& layer : layers) {
                architecture.push_back(layer.getWeights().size());
            }
        }
        
        size_t num_layers = architecture.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(architecture.data()), num_layers * sizeof(int));
        
        for (const auto& layer : layers) {
            const auto& weights = layer.getWeights();
            for (const auto& row : weights) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
            
            const auto& biases = layer.getBiases();
            file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
        }
    }
    
    void loadModel(const string& filename) {
        ifstream file(filename, ios::binary);
        if (!file) throw runtime_error("Cannot open file for reading");
        
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
        
        vector<int> architecture(num_layers);
        file.read(reinterpret_cast<char*>(architecture.data()), num_layers * sizeof(int));
        
        vector<Activation> activations;
        for (size_t i = 1; i < num_layers; ++i) {
            activations.push_back(i == num_layers - 1 ? Activations::Softmax() : Activations::Sigmoid());
        }
        
        *this = ParallelMLP(architecture, activations, batch_size, num_threads);
        
        for (auto& layer : layers) {
            auto& weights = layer.getWeights();
            for (auto& row : weights) {
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            
            auto& biases = layer.getBiases();
            file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
        }
    }
};

// ==================== MNIST DATASET LOADER ====================
class MNISTDataset {
public:
    static Matrix load_images(const string& filename, int max_images = -1);
    static Matrix load_labels(const string& filename, int max_labels = -1);
    
private:
    static int32_t read_int(ifstream& file) {
        int32_t value;
        file.read(reinterpret_cast<char*>(&value), 4);
        return __builtin_bswap32(value);
    }
};

Matrix MNISTDataset::load_images(const string& filename, int max_images) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open MNIST images file: " + filename);

    if (read_int(file) != 2051) throw runtime_error("Invalid MNIST image file");
    
    int num_images = read_int(file);
    int rows = read_int(file);
    int cols = read_int(file);

    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }

    Matrix images(num_images, Vector(rows * cols));
    
    for (auto& image : images) {
        for (auto& pixel : image) {
            uint8_t val;
            file.read(reinterpret_cast<char*>(&val), 1);
            pixel = val / 255.0f;
        }
    }

    return images;
}

Matrix MNISTDataset::load_labels(const string& filename, int max_labels) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open MNIST labels file: " + filename);

    if (read_int(file) != 2049) throw runtime_error("Invalid MNIST label file");
    
    int num_labels = read_int(file);
    if (max_labels > 0 && max_labels < num_labels) {
        num_labels = max_labels;
    }

    Matrix labels(num_labels, Vector(10, 0.0f));
    
    for (auto& label : labels) {
        uint8_t val;
        file.read(reinterpret_cast<char*>(&val), 1);
        label[val] = 1.0f;
    }

    return labels;
}

// ==================== MAIN ====================
int main() {
    const string dataset_path = "../dataset/mnist/";
    const int training_samples = 60000;
    const int test_samples = 10000;
    cout << "Loading MNIST data..." << endl;
    auto train_images = MNISTDataset::load_images(dataset_path + "train-images.idx3-ubyte", training_samples);
    auto train_labels = MNISTDataset::load_labels(dataset_path + "train-labels.idx1-ubyte", training_samples);
    auto test_images = MNISTDataset::load_images(dataset_path + "t10k-images.idx3-ubyte", test_samples);
    auto test_labels = MNISTDataset::load_labels(dataset_path + "t10k-labels.idx1-ubyte", test_samples);
    cout << "Data loaded successfully!" << endl;

    // Crear red neuronal con Adam optimizer
    ParallelMLP mlp({784, 256, 128, 10}, 
                   {Activations::Sigmoid(), Activations::Sigmoid(), Activations::Softmax()},
                   64, 4, "adam", 0.001f);
    
    // Entrenamiento
    mlp.trainParallel(train_images, train_labels, 15, "training_metrics.csv");
    
    // Evaluación
    float accuracy = mlp.evaluate(test_images, test_labels);
    cout << "Test Accuracy: " << accuracy << "%\n";
    
    // Guardar modelo
    mlp.saveModel("mnist_model.bin");
    
    return 0;
}