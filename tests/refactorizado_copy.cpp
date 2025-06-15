#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string>
#include <omp.h>

using namespace std;

// Funciones de activación y derivadas
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) {
    return (x > 0) ? x : 0;
}

double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

double tanh_derivative(double x) {
    return 1 - tanh(x) * tanh(x);
}

// Función de pérdida
double cross_entropy(const vector<double>& output, const vector<double>& target) {
    double loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        loss -= target[i] * log(output[i] + 1e-15);
    }
    return loss;
}

// Softmax
vector<double> softmax(const vector<double>& input) {
    vector<double> result(input.size());
    double max_val = *max_element(input.begin(), input.end());
    double sum = 0.0;
    
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = exp(input[i] - max_val);
        sum += result[i];
    }
    
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

// Clase base para optimizadores
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update_weights(vector<vector<double>>& weights, 
                              vector<vector<double>>& d_weights,
                              vector<double>& biases,
                              vector<double>& d_biases,
                              double learning_rate,
                              size_t layer_idx = 0) = 0;
    virtual string get_name() const = 0;
    virtual void initialize_for_layer(size_t layer_idx, size_t neurons, size_t inputs) {}
};

// SGD (Descenso de Gradiente Estocástico)
class SGD : public Optimizer {
public:
    void update_weights(vector<vector<double>>& weights, 
                       vector<vector<double>>& d_weights,
                       vector<double>& biases,
                       vector<double>& d_biases,
                       double learning_rate,
                       size_t layer_idx = 0) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learning_rate * d_weights[i][j];
            }
            biases[i] -= learning_rate * d_biases[i];
        }
    }
    
    string get_name() const override { return "SGD"; }
};

// RMSprop
class RMSprop : public Optimizer {
private:
    double epsilon = 1e-8;
    double gamma = 0.9;
    vector<vector<vector<double>>> s_weights;
    vector<vector<double>> s_biases;
    
public:
    void initialize_for_layer(size_t layer_idx, size_t neurons, size_t inputs) override {
        if (layer_idx >= s_weights.size()) {
            s_weights.resize(layer_idx + 1);
            s_biases.resize(layer_idx + 1);
        }
        
        s_weights[layer_idx].resize(neurons, vector<double>(inputs, 0.0));
        s_biases[layer_idx].resize(neurons, 0.0);
    }
    
    void update_weights(vector<vector<double>>& weights, 
                       vector<vector<double>>& d_weights,
                       vector<double>& biases,
                       vector<double>& d_biases,
                       double learning_rate,
                       size_t layer_idx = 0) override {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                s_weights[layer_idx][i][j] = gamma * s_weights[layer_idx][i][j] + 
                                           (1 - gamma) * d_weights[i][j] * d_weights[i][j];
                weights[i][j] -= learning_rate * d_weights[i][j] / 
                               (sqrt(s_weights[layer_idx][i][j]) + epsilon);
            }
            
            s_biases[layer_idx][i] = gamma * s_biases[layer_idx][i] + 
                                    (1 - gamma) * d_biases[i] * d_biases[i];
            biases[i] -= learning_rate * d_biases[i] / 
                       (sqrt(s_biases[layer_idx][i]) + epsilon);
        }
    }
    
    string get_name() const override { return "RMSprop"; }
};

// Adam
class Adam : public Optimizer {
private:
    double epsilon = 1e-8;
    double beta1 = 0.9;
    double beta2 = 0.999;
    vector<vector<vector<double>>> m_weights;
    vector<vector<vector<double>>> v_weights;
    vector<vector<double>> m_biases;
    vector<vector<double>> v_biases;
    int t = 0;
    
public:
    void initialize_for_layer(size_t layer_idx, size_t neurons, size_t inputs) override {
        if (layer_idx >= m_weights.size()) {
            m_weights.resize(layer_idx + 1);
            v_weights.resize(layer_idx + 1);
            m_biases.resize(layer_idx + 1);
            v_biases.resize(layer_idx + 1);
        }
        
        m_weights[layer_idx].resize(neurons, vector<double>(inputs, 0.0));
        v_weights[layer_idx].resize(neurons, vector<double>(inputs, 0.0));
        m_biases[layer_idx].resize(neurons, 0.0);
        v_biases[layer_idx].resize(neurons, 0.0);
    }
    
    void update_weights(vector<vector<double>>& weights, 
                       vector<vector<double>>& d_weights,
                       vector<double>& biases,
                       vector<double>& d_biases,
                       double learning_rate,
                       size_t layer_idx = 0) override {
        t++;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                m_weights[layer_idx][i][j] = beta1 * m_weights[layer_idx][i][j] + 
                                           (1 - beta1) * d_weights[i][j];
                v_weights[layer_idx][i][j] = beta2 * v_weights[layer_idx][i][j] + 
                                           (1 - beta2) * d_weights[i][j] * d_weights[i][j];
                
                double m_hat = m_weights[layer_idx][i][j] / (1 - pow(beta1, t));
                double v_hat = v_weights[layer_idx][i][j] / (1 - pow(beta2, t));
                
                weights[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            
            m_biases[layer_idx][i] = beta1 * m_biases[layer_idx][i] + 
                                    (1 - beta1) * d_biases[i];
            v_biases[layer_idx][i] = beta2 * v_biases[layer_idx][i] + 
                                    (1 - beta2) * d_biases[i] * d_biases[i];
            
            double m_hat = m_biases[layer_idx][i] / (1 - pow(beta1, t));
            double v_hat = v_biases[layer_idx][i] / (1 - pow(beta2, t));
            
            biases[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
    
    string get_name() const override { return "Adam"; }
};

// Capa de la red neuronal
class Layer {
public:
    enum Activation { SIGMOID, RELU, TANH, SOFTMAX, LINEAR };
    
    Layer(size_t input_size, size_t output_size, Activation activation, bool is_output = false)
        : input_size(input_size), output_size(output_size), activation(activation), is_output(is_output) {
        initialize_weights();
    }
    
    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        double range = (activation == RELU) ? sqrt(2.0 / input_size) : sqrt(1.0 / input_size);
        normal_distribution<double> dist(0.0, range);
        
        weights.resize(output_size, vector<double>(input_size));
        biases.resize(output_size, 0.0);
        
        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                weights[i][j] = dist(gen);
            }
        }
    }
    
    vector<double> forward(const vector<double>& input) {
        this->input = input;
        output.resize(output_size);
        
        for (size_t i = 0; i < output_size; ++i) {
            output[i] = biases[i];
            for (size_t j = 0; j < input_size; ++j) {
                output[i] += weights[i][j] * input[j];
            }
        }
        
        activated_output = output;
        if (activation == SIGMOID) {
            for (auto& val : activated_output) val = sigmoid(val);
        } else if (activation == RELU) {
            for (auto& val : activated_output) val = relu(val);
        } else if (activation == TANH) {
            for (auto& val : activated_output) val = tanh(val);
        } else if (activation == SOFTMAX && is_output) {
            activated_output = softmax(output);
        }
        
        return activated_output;
    }
    
    vector<double> backward(const vector<double>& output_error, 
                          double learning_rate, 
                          Optimizer* optimizer,
                          size_t layer_idx) {
        vector<double> input_error(input_size, 0.0);
        vector<vector<double>> d_weights(output_size, vector<double>(input_size, 0.0));
        vector<double> d_biases(output_size, 0.0);
        
        vector<double> activation_derivative(output_size);
        if (activation == SIGMOID) {
            for (size_t i = 0; i < output_size; ++i) {
                activation_derivative[i] = sigmoid_derivative(output[i]);
            }
        } else if (activation == RELU) {
            for (size_t i = 0; i < output_size; ++i) {
                activation_derivative[i] = relu_derivative(output[i]);
            }
        } else if (activation == TANH) {
            for (size_t i = 0; i < output_size; ++i) {
                activation_derivative[i] = tanh_derivative(output[i]);
            }
        } else {
            for (size_t i = 0; i < output_size; ++i) {
                activation_derivative[i] = 1.0;
            }
        }
        
        for (size_t i = 0; i < output_size; ++i) {
            double delta = output_error[i] * activation_derivative[i];
            d_biases[i] = delta;
            
            for (size_t j = 0; j < input_size; ++j) {
                d_weights[i][j] = delta * input[j];
                input_error[j] += delta * weights[i][j];
            }
        }
        
        optimizer->update_weights(weights, d_weights, biases, d_biases, learning_rate, layer_idx);
        
        return input_error;
    }
    
    size_t get_input_size() const { return input_size; }
    size_t get_output_size() const { return output_size; }
    
private:
    size_t input_size, output_size;
    Activation activation;
    bool is_output;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> input, output, activated_output;
};

// MLP (Perceptrón Multicapa)
class MLP {
public:
    MLP(size_t input_size, const vector<size_t>& hidden_layers, size_t output_size, 
        Layer::Activation hidden_activation = Layer::RELU, 
        Layer::Activation output_activation = Layer::SOFTMAX) {
        size_t prev_size = input_size;
        for (size_t neurons : hidden_layers) {
            layers.emplace_back(prev_size, neurons, hidden_activation);
            prev_size = neurons;
        }
        layers.emplace_back(prev_size, output_size, output_activation, true);
    }
    
    void set_optimizer(const string& optimizer_name) {
        if (optimizer_name == "sgd") {
            optimizer = make_unique<SGD>();
        } else if (optimizer_name == "rmsprop") {
            optimizer = make_unique<RMSprop>();
        } else if (optimizer_name == "adam") {
            optimizer = make_unique<Adam>();
        } else {
            throw invalid_argument("Optimizador desconocido");
        }
        initialize_optimizer();
    }
    
    vector<double> predict(const vector<double>& input) {
        vector<double> output = input;
        for (auto& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
    
    double calculate_accuracy(const vector<vector<double>>& inputs, 
                            const vector<vector<double>>& targets) {
        size_t correct = 0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            vector<double> output = predict(inputs[i]);
            size_t predicted = max_element(output.begin(), output.end()) - output.begin();
            size_t actual = max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();
            
            if (predicted == actual) {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / inputs.size();
    }
    
    void train_model(const vector<vector<double>>& train_inputs,
                   const vector<vector<double>>& train_targets,
                   const vector<vector<double>>& test_inputs,
                   const vector<vector<double>>& test_targets,
                   int epochs,
                   double learning_rate,
                   const string& log_filename = "training_log.csv",
                   int verbose_interval = 10) {
        
        ofstream log_file(log_filename);
        if (!log_file.is_open()) {
            throw runtime_error("No se pudo abrir el archivo de log");
        }
        
        log_file << "epoch,train_loss,train_accuracy,test_loss,test_accuracy,time_ms\n";
        
        cout << "Comenzando entrenamiento con " << optimizer->get_name() << "\n";
        cout << "Epochs: " << epochs << ", Tasa de aprendizaje: " << learning_rate << "\n";
        cout << "Guardando logs en: " << log_filename << "\n\n";
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto start_time = chrono::high_resolution_clock::now();
            
            double total_loss = 0.0;
            
            // Entrenamiento
            for (size_t i = 0; i < train_inputs.size(); ++i) {
                total_loss += train(train_inputs[i], train_targets[i], learning_rate);
            }
            double avg_train_loss = total_loss / train_inputs.size();
            double train_accuracy = calculate_accuracy(train_inputs, train_targets);
            
            // Evaluación en test
            double test_loss = 0.0;
            for (size_t i = 0; i < test_inputs.size(); ++i) {
                vector<double> output = predict(test_inputs[i]);
                test_loss += cross_entropy(output, test_targets[i]);
            }
            double avg_test_loss = test_loss / test_inputs.size();
            double test_accuracy = calculate_accuracy(test_inputs, test_targets);
            
            auto end_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            
            // Escribir en el log
            log_file << epoch << "," 
                    << avg_train_loss << "," 
                    << train_accuracy << "," 
                    << avg_test_loss << "," 
                    << test_accuracy << ","
                    << duration.count() << "\n";
            
            // Mostrar progreso
            if (epoch % verbose_interval == 0 || epoch == epochs - 1) {
                cout << "Epoch " << setw(4) << epoch + 1<< " - "
                      << "Train Loss: " << setw(10) << setprecision(6) << avg_train_loss
                      << " Acc: " << setw(6) << setprecision(4) << train_accuracy * 100 << "% | "
                      << "Test Loss: " << setw(10) << setprecision(6) << avg_test_loss
                      << " Acc: " << setw(6) << setprecision(4) << test_accuracy * 100 << "%"
                      << " Time: " << duration.count() << "ms\n";
            }
        }
        
        log_file.close();
        cout << "\nEntrenamiento completado. Logs guardados en " << log_filename << "\n";
    }
    
private:
    vector<Layer> layers;
    unique_ptr<Optimizer> optimizer = make_unique<SGD>();
    
    void initialize_optimizer() {
        for (size_t i = 0; i < layers.size(); ++i) {
            optimizer->initialize_for_layer(i, layers[i].get_output_size(), 
                                         layers[i].get_input_size());
        }
    }
    
    double train(const vector<double>& input, 
                const vector<double>& target, 
                double learning_rate) {
        vector<double> output = predict(input);
        vector<double> output_error(output.size());
        
        for (size_t i = 0; i < output.size(); ++i) {
            output_error[i] = output[i] - target[i];
        }
        
        double loss = cross_entropy(output, target);
        
        vector<double> error = output_error;
        for (int i = layers.size() - 1; i >= 0; --i) {
            error = layers[i].backward(error, learning_rate, optimizer.get(), i);
        }
        
        return loss;
    }
};

// Función para imprimir un vector
void print_vector(const vector<double>& vec) {
    cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        cout << fixed << setprecision(4) << vec[i];
        if (i < vec.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}


// Clase para manejar datos MNIST con double
class MNISTDataset {
public:
    static vector<vector<double>> loadImages(const string& filename, int max_images = -1);
    static vector<vector<double>> loadLabels(const string& filename, int max_labels = -1);
    static void displayImage(const vector<double>& image, int rows = 28, int cols = 28);
};

// -------------------- MNIST Dataset --------------------
vector<vector<double>> MNISTDataset::loadImages(const string& filename, int max_images) {
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

    vector<vector<double>> images;
    images.reserve(num_images);

    for (int i = 0; i < num_images; ++i) {
        vector<double> image(rows * cols);
        for (int j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = pixel / 255.0; // Normalizar a [0,1] con double
        }
        images.push_back(move(image));
    }

    return images;
}

vector<vector<double>> MNISTDataset::loadLabels(const string& filename, int max_labels) {
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

    vector<vector<double>> labels;
    labels.reserve(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        
        vector<double> one_hot(10, 0.0);
        one_hot[label] = 1.0;
        labels.push_back(move(one_hot));
    }

    return labels;
}

void MNISTDataset::displayImage(const vector<double>& image, int rows, int cols) {
    const string shades = " .:-=+*#%@";
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double pixel = image[i * cols + j];
            int level = static_cast<int>(pixel * (shades.size() - 1));
            cout << shades[level] << shades[level];
        }
        cout << endl;
    }
}

int main() {
    const string dataset_path = "../dataset/mnist/"; // Cambiar si usas Fashion-MNIST
    const string prefix = "mnist";

    const int training_samples = 60000;
    const int test_samples = 10000;

    const size_t input_size = 28 * 28;
    const vector<size_t> hidden_layers = {64, 32};
    const size_t output_size = 10;

    const int epochs = 20;
    const double learning_rate = 0.001;
    const int print_interval = 1;

    try {
        cout << "Cargando dataset desde: " << dataset_path << endl;
        auto train_images = MNISTDataset::loadImages(dataset_path + "train-images.idx3-ubyte", training_samples);
        auto train_labels = MNISTDataset::loadLabels(dataset_path + "train-labels.idx1-ubyte", training_samples);
        auto test_images  = MNISTDataset::loadImages(dataset_path + "t10k-images.idx3-ubyte", test_samples);
        auto test_labels  = MNISTDataset::loadLabels(dataset_path + "t10k-labels.idx1-ubyte", test_samples);

        vector<string> optimizers = {"adam", "rmsprop", "sgd"};

        for (const auto& opt : optimizers) {
            cout << "\n====================================" << endl;
            cout << "🧠 Entrenando modelo con optimizador: " << opt << endl;
            cout << "📥 Entrada: " << input_size << " dimensiones" << endl;
            cout << "🧱 Capas ocultas: ";
            for (auto h : hidden_layers) cout << h << " ";
            cout << "\n🎯 Salida: " << output_size << " clases" << endl;
            cout << "🔁 Épocas: " << epochs << ", 📈 Learning rate: " << learning_rate << endl;

            // Crear red neuronal
            MLP mlp(input_size, hidden_layers, output_size, Layer::RELU, Layer::SOFTMAX);
            mlp.set_optimizer(opt);

            // Nombre de archivo log
            string log_file = prefix + "_" + opt + "_log.csv";
            cout << "📄 Guardando log de entrenamiento en: " << log_file << endl;

            // Entrenar modelo
            mlp.train_model(train_images, train_labels,
                            test_images, test_labels,
                            epochs, learning_rate,
                            log_file, print_interval);

            // Evaluar precisión final
            double test_acc = mlp.calculate_accuracy(test_images, test_labels);
            cout << "✅ Precisión final en test con " << opt << ": " << test_acc * 100 << "%" << endl;
        }

    } catch (const exception& e) {
        cerr << "❌ Error durante ejecución: " << e.what() << endl;
        return 1;
    }

    cout << "\n🎉 Entrenamiento finalizado para todos los optimizadores." << endl;
    return 0;
}