#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
#include <thread>
#include <mutex>

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;

// ==================== DECLARACIONES ====================

using Matrix = std::vector<std::vector<float>>;
using Vector = std::vector<float>;

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
        // Softmax es un caso especial, no necesita función de derivada aquí
        // porque la derivada ya se maneja directamente en el backward pass
        return Activation(
            nullptr,  // No usamos activate para softmax
            nullptr   // La derivada se maneja directamente
        );
    }
}

class Layer {
private:
    Matrix weights;  // weights[l][k] = peso de la neurona k a la neurona l
    Vector biases;
    Vector outputs;
    Vector deltas;
    Vector z;  // net_inputs
    Activation activation;
    bool is_output = false;
    bool use_softmax = false;
    
public:
    Layer(int input_size, int output_size, Activation act, bool output_layer = false) 
        : activation(act), is_output(output_layer) {
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
    }
    
    void setUseSoftmax(bool use) { use_softmax = use; }
    
    // Forward pass para un batch de inputs
    Matrix forward(const Matrix& inputs) {
        Matrix batch_outputs(inputs.size(), Vector(weights.size())); // outputs.size() -> weights.size()
        
        // Paralelizar este bucle externo sería fácil con OpenMP
        for (size_t b = 0; b < inputs.size(); ++b) {
            // Multiplicación matriz-vector: z = W * x + b
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
            if (b >= next_layer_deltas.size()) break;  // Prevent out-of-bounds
            
            for (size_t k = 0; k < deltas.size(); ++k) {
                float sum = 0.0f;
                for (size_t l = 0; l < next_layer_weights.size(); ++l) {
                    if (l < next_layer_deltas[b].size()) {  // Check bounds
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
                    // Para softmax, la derivada ya está considerada en la resta
                    deltas[k] = outputs[k] - targets[b][k];
                } else {
                    deltas[k] = (outputs[k] - targets[b][k]) * activation.derive(outputs[k]);
                }
            }
        }
    }
    
    // Actualizar pesos
    void updateWeights(const Matrix& inputs, float learning_rate) {
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t k = 0; k < weights[l].size(); ++k) {
                float gradient = 0.0f;
                for (size_t b = 0; b < inputs.size(); ++b) {
                    gradient += deltas[l] * inputs[b][k];
                }
                weights[l][k] -= learning_rate * gradient / inputs.size();
            }
            
            // Actualizar bias
            float bias_gradient = 0.0f;
            for (size_t b = 0; b < inputs.size(); ++b) {
                bias_gradient += deltas[l];
            }
            biases[l] -= learning_rate * bias_gradient / inputs.size();
        }
    }
    
    const Vector& getOutputs() const { return outputs; }
    const Vector& getDeltas() const { return deltas; }
    const Matrix& getWeights() const { return weights; }
    const Vector& getBiases() const { return biases; }

    Matrix& getWeights() { return weights; }
    Vector& getBiases() { return biases; }
    Matrix getBatchOutputs() const { return {outputs}; }  // Para convertir a Matrix (batch de 1)
//    const Vector& getOutputs() const { return outputs; }
};

class ParallelMLP {
private:
    vector<Layer> layers;
    int batch_size;
    int num_threads;
    
public:
    ParallelMLP(const vector<int>& architecture, 
               const vector<Activation>& activations,
               int batch_size = 32, 
               int threads = thread::hardware_concurrency())
        : batch_size(batch_size), num_threads(threads) {
        
        for (size_t i = 1; i < architecture.size(); ++i) {
            bool is_output = (i == architecture.size() - 1);
            layers.emplace_back(architecture[i-1], architecture[i], 
                               activations[i-1], is_output);
            
            if (is_output && activations[i-1].activate == nullptr) {
                layers.back().setUseSoftmax(true);
            }
        }

        // En el constructor de ParallelMLP
        if (architecture.size() < 2 || architecture.size() != activations.size() + 1) {
            throw runtime_error("Invalid architecture or activations size");
        }
    }
    
    // Forward pass para un batch
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
    
    // Backward pass
    void backward(const Matrix& inputs, const Matrix& targets) {
        // Forward pass primero para calcular todas las activaciones
        forward(inputs);
        
        // Backward para capa de salida
        if (layers.empty()) return;
        
        // Asegúrate de que las dimensiones son correctas
        const Matrix& last_hidden_output = layers.size() > 1 ? 
            Matrix{layers[layers.size()-2].getOutputs()} : inputs;
        
        layers.back().backwardOutput(last_hidden_output, targets);
        
        // Backward para capas ocultas
        for (int i = layers.size() - 2; i >= 0; --i) {
            const Matrix& prev_outputs = i > 0 ? 
                Matrix{layers[i-1].getOutputs()} : inputs;
                
            layers[i].backward(
                prev_outputs,
                Matrix{layers[i+1].getDeltas()},
                layers[i+1].getWeights()
            );
        }
        
        // Actualizar pesos
        for (size_t i = 0; i < layers.size(); ++i) {
            const Matrix& layer_inputs = (i == 0) ? inputs : Matrix{layers[i-1].getOutputs()};
            layers[i].updateWeights(layer_inputs, LEARNING_RATE);
        }
    }
    
    // Entrenamiento paralelizado
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
            
            // Procesar en batches
            for (size_t b = 0; b < all_inputs.size(); b += batch_size) {
                size_t batch_end = min(b + batch_size, all_inputs.size());

                if (batch_end > all_inputs.size() || batch_end > all_targets.size()) {
                    throw runtime_error("Batch access out of range");
                }
                Matrix batch_inputs(all_inputs.begin() + b, all_inputs.begin() + batch_end);
                Matrix batch_targets(all_targets.begin() + b, all_targets.begin() + batch_end);
                
                // Forward + backward
                backward(batch_inputs, batch_targets);
                
                // Calcular métricas
                auto outputs = forward(batch_inputs);
                for (size_t i = 0; i < outputs.size(); ++i) {
                    // Pérdida
                    for (size_t j = 0; j < outputs[i].size(); ++j) {
                        total_loss += 0.5f * pow(outputs[i][j] - batch_targets[i][j], 2);
                    }
                    
                    // Precisión
                    int predicted = distance(outputs[i].begin(), 
                                          max_element(outputs[i].begin(), outputs[i].end()));
                    int actual = distance(batch_targets[i].begin(), 
                                       max_element(batch_targets[i].begin(), batch_targets[i].end()));
                    if (predicted == actual) correct++;
                }
            }
            
            // Mostrar estadísticas
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
    
    // Evaluación
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
    
    // Guardar modelo
    void saveModel(const string& filename) {
        ofstream file(filename, ios::binary);
        if (!file) throw runtime_error("Cannot open file for writing");
        
        // Guardar arquitectura
        vector<int> architecture;
        if (!layers.empty()) {
            architecture.push_back(layers[0].getWeights()[0].size()); // input size
            for (const auto& layer : layers) {
                architecture.push_back(layer.getWeights().size()); // output size
            }
        }
        
        size_t num_layers = architecture.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(architecture.data()), num_layers * sizeof(int));
        
        // Guardar pesos
        for (const auto& layer : layers) {
            const auto& weights = layer.getWeights();
            for (const auto& row : weights) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
            
            // Guardar biases
            const auto& biases = layer.getBiases();
            file.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
        }
    }
    
    // Cargar modelo
    void loadModel(const string& filename) {
        ifstream file(filename, ios::binary);
        if (!file) throw runtime_error("Cannot open file for reading");
        
        // Leer arquitectura
        size_t num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));
        
        vector<int> architecture(num_layers);
        file.read(reinterpret_cast<char*>(architecture.data()), num_layers * sizeof(int));
        
        // Recrear capas (asumiendo que las activaciones son las mismas)
        vector<Activation> activations;
        for (size_t i = 1; i < num_layers; ++i) {
            activations.push_back(i == num_layers - 1 ? Activations::Softmax() : Activations::Sigmoid());
        }
        
        *this = ParallelMLP(architecture, activations, batch_size, num_threads);
        
        // Cargar pesos
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

// Clase para manejar datos MNIST (similar a la original)
class MNISTDataset {
public:
    static vector<vector<float>> loadImages(const string& filename, int max_images = -1);
    static vector<vector<float>> loadLabels(const string& filename, int max_labels = -1);
};

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

// Ejemplo de uso
int main() {
    const string dataset_path = "../dataset/mnist/"; // Cambia a "../dataset/fashion-mnist/" si quieres
    // Cargar datos MNIST
    const int training_samples = 60000;
    const int test_samples = 10000;
    cout << "Loading MNIST data..." << endl;
auto train_images = MNISTDataset::loadImages(dataset_path + "train-images.idx3-ubyte", training_samples);
auto train_labels = MNISTDataset::loadLabels(dataset_path + "train-labels.idx1-ubyte", training_samples);
auto test_images = MNISTDataset::loadImages(dataset_path + "t10k-images.idx3-ubyte", test_samples);
auto test_labels = MNISTDataset::loadLabels(dataset_path + "t10k-labels.idx1-ubyte", test_samples);
cout << "Data loaded successfully!" << endl;

    // Crear red neuronal
    ParallelMLP mlp({784, 256, 128, 10}, 
                   {Activations::Sigmoid(), Activations::Sigmoid(), Activations::Softmax()},
                   64, 4); // batch_size=64, 4 threads
    
    
    // Entrenamiento
    mlp.trainParallel(train_images, train_labels, 15, "training_metrics.csv");
    
    // Evaluación
    float accuracy = mlp.evaluate(test_images, test_labels);
    cout << "Test Accuracy: " << accuracy << "%\n";
    
    // Guardar modelo
    mlp.saveModel("mnist_model.bin");
    
    return 0;
}