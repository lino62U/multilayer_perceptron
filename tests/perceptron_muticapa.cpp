#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <numeric>  // Para std::accumulate en softmax

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;

enum FuncionActivacion {
    SIGMOIDE,
    RELU,
    TANH,
    SOFTMAX
};

struct Neurona {
    float salida = 0.0f;
    float net = 0.0f;
    float delta = 0.0f;
    bool esBias = false;
    bool esSalida = false;
    FuncionActivacion activacion = SIGMOIDE;  // Por defecto sigmoide

    vector<pair<Neurona*, float*>> entradas;
    vector<pair<Neurona*, float*>> salidas;

    // Funciones de activación
    float sigmoide(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    float derivada_sigmoide(float y) {
        return y * (1.0f - y);
    }

    float relu(float x) {
        return x > 0 ? x : 0.0f;
    }

    float derivada_relu(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }

    float tanh_act(float x) {
        return tanh(x);
    }

    float derivada_tanh(float y) {
        return 1.0f - y * y;
    }

    // Softmax se calcula a nivel de capa, no de neurona individual
    float activar(float x) {
        switch(activacion) {
            case SIGMOIDE: return sigmoide(x);
            case RELU: return relu(x);
            case TANH: return tanh_act(x);
            default: return sigmoide(x); // SOFTMAX se maneja aparte
        }
    }

    float derivada(float y) {
        switch(activacion) {
            case SIGMOIDE: return derivada_sigmoide(y);
            case RELU: return derivada_relu(y);
            case TANH: return derivada_tanh(y);
            default: return derivada_sigmoide(y); // SOFTMAX se maneja aparte
        }
    }

    void calcularSalida() {
        if (esBias) {
            salida = 1.0f;
            return;
        }

        float suma = 0.0f;
        for (auto& entrada : entradas) {
            suma += *(entrada.second) * entrada.first->salida;
        }
        net = suma;
        salida = activar(suma);
    }

    void calcularDeltaSalida(float objetivo) {
        delta = (salida - objetivo) * derivada(salida);
    }

    void calcularDeltaOculta() {
        float suma = 0.0f;
        for (auto& s : salidas) {
            suma += *(s.second) * s.first->delta;
        }
        delta = derivada(salida) * suma;
    }

    void actualizarPesos() {
        if (esBias) return;
        for (auto& entrada : entradas) {
            *(entrada.second) -= LEARNING_RATE * delta * entrada.first->salida;
        }
    }
};

class PerceptronMulticapa {
    vector<vector<Neurona*>> capas;
    vector<FuncionActivacion> funcionesActivacion;

public:
    void crearRed(const vector<int>& estructura, const vector<FuncionActivacion>& funciones) {
        srand(RANDOM_SEED);
        funcionesActivacion = funciones;

        for (int i = 0; i < estructura.size(); ++i) {
            vector<Neurona*> capa;

            int n_neuronas = estructura[i];
            if (i != estructura.size() - 1) n_neuronas += 1;  // Agregar bias si no es capa de salida

            for (int j = 0; j < n_neuronas; ++j) {
                Neurona* n = new Neurona();
                if (i != estructura.size() - 1 && j == n_neuronas - 1) {
                    n->esBias = true;
                    n->salida = 1.0f;
                }
                n->esSalida = (i == estructura.size() - 1);
                n->activacion = (i < funciones.size()) ? funciones[i] : SIGMOIDE;
                capa.push_back(n);
            }

            capas.push_back(capa);
        }

        // Conectar neuronas capa a capa
        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                if (neurona->esBias) continue;
                for (auto& anterior : capas[i - 1]) {
                    float* peso = new float(((rand() % 100) / 100.0f) - 0.5f);
                    neurona->entradas.push_back({anterior, peso});
                    anterior->salidas.push_back({neurona, peso});
                }
            }
        }
    }

    void aplicarSoftmax() {
        if (funcionesActivacion.back() != SOFTMAX) return;
        
        vector<Neurona*>& capaSalida = capas.back();
        float suma = 0.0f;
        
        // Calcular exponenciales y suma
        for (auto& neurona : capaSalida) {
            if (neurona->esBias) continue;
            neurona->salida = exp(neurona->net);
            suma += neurona->salida;
        }
        
        // Normalizar
        for (auto& neurona : capaSalida) {
            if (neurona->esBias) continue;
            neurona->salida /= suma;
        }
    }

    void establecerEntrada(const vector<float>& entradas) {
        for (int i = 0; i < entradas.size(); ++i) {
            capas[0][i]->salida = entradas[i];
        }
    }

    vector<float> ejecutarRed() {
        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                if (!neurona->esBias) {
                    neurona->calcularSalida();
                }
            }
            
            // Aplicar softmax si es la última capa y está configurado
            if (i == capas.size() - 1 && funcionesActivacion.back() == SOFTMAX) {
                aplicarSoftmax();
            }
        }

        vector<float> salidas;
        for (auto& neurona : capas.back()) {
            if (!neurona->esBias) {
                salidas.push_back(neurona->salida);
            }
        }
        return salidas;
    }

    void backpropagation(const vector<float>& objetivos) {
        // Capa de salida (manejo especial para softmax)
        if (funcionesActivacion.back() == SOFTMAX) {
            vector<Neurona*>& capaSalida = capas.back();
            for (int i = 0; i < capaSalida.size(); ++i) {
                if (!capaSalida[i]->esBias) {
                    capaSalida[i]->delta = capaSalida[i]->salida - objetivos[i];
                }
            }
        } else {
            for (int i = 0; i < capas.back().size(); ++i) {
                if (!capas.back()[i]->esBias) {
                    capas.back()[i]->calcularDeltaSalida(objetivos[i]);
                }
            }
        }

        // Capas ocultas
        for (int i = capas.size() - 2; i > 0; --i) {
            for (auto& neurona : capas[i]) {
                if (!neurona->esBias)
                    neurona->calcularDeltaOculta();
            }
        }

        // Actualizar pesos
        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                neurona->actualizarPesos();
            }
        }
    }

    void entrenar(const vector<float>& entrada, const vector<float>& objetivo, int epocas = 1000) {
        for (int i = 0; i < epocas; ++i) {
            establecerEntrada(entrada);
            ejecutarRed();
            backpropagation(objetivo);
        }
    }

    void imprimirRed() {
        for (int i = 0; i < capas.size(); ++i) {
            cout << "Capa " << i << " (";
            switch(funcionesActivacion[i]) {
                case SIGMOIDE: cout << "Sigmoide"; break;
                case RELU: cout << "ReLU"; break;
                case TANH: cout << "Tanh"; break;
                case SOFTMAX: cout << "Softmax"; break;
            }
            cout << "):\n";
            
            for (auto& neurona : capas[i]) {
                cout << "  Neurona ";
                if (neurona->esBias) {
                    cout << "[bias] ";
                }
                cout << "Salida: " << neurona->salida << ", Delta: " << neurona->delta << endl;

                if (!neurona->entradas.empty()) {
                    cout << "    Pesos de entrada:\n";
                    for (auto& entrada : neurona->entradas) {
                        cout << "      Desde neurona "
                             << (entrada.first->esBias ? "[bias]" : "") 
                             << " salida=" << entrada.first->salida
                             << " -> peso=" << *(entrada.second) << endl;
                    }
                }
            }
        }
    }

    void guardarPesos(const string& archivo) {
        ofstream f(archivo, ios::binary);
        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                for (auto& entrada : neurona->entradas) {
                    f.write((char*)entrada.second, sizeof(float));
                }
            }
        }
    }

    void cargarPesos(const string& archivo) {
        ifstream f(archivo, ios::binary);
        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                for (auto& entrada : neurona->entradas) {
                    f.read((char*)entrada.second, sizeof(float));
                }
            }
        }
    }
};

// [Resto del código permanece igual: leerDatos, leerImagenesMNIST, leerEtiquetasMNIST, 
//  mostrarImagenConsola, calcularLoss, entrenarModelo, probarModelo, parseEstructuraRed]

// Función para leer los datos desde un archivo
vector<vector<float>> leerDatos(const string& archivo) {
    ifstream archivoEntrada(archivo);
    vector<vector<float>> datos;

    float x, y;
    while (archivoEntrada >> x >> y) {
        datos.push_back({x, y});
    }

    return datos;
}

#include <fstream>
#include <vector>
#include <cstdint>

using namespace std;

vector<vector<float>> leerImagenesMNIST(const string& archivo, int cantidad) {
    ifstream f(archivo, ios::binary);
    if (!f) throw runtime_error("No se pudo abrir archivo de imágenes");

    int32_t magic, num, rows, cols;
    f.read((char*)&magic, 4);
    f.read((char*)&num, 4);
    f.read((char*)&rows, 4);
    f.read((char*)&cols, 4);

    magic = __builtin_bswap32(magic);
    num = __builtin_bswap32(num);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    vector<vector<float>> imagenes;
    for (int i = 0; i < cantidad; ++i) {
        vector<float> imagen(rows * cols);
        for (int j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            f.read((char*)&pixel, 1);
            imagen[j] = pixel / 255.0f;  // Normalizado
        }
        imagenes.push_back(imagen);
    }
    return imagenes;
}

vector<vector<float>> leerEtiquetasMNIST(const string& archivo, int cantidad) {
    ifstream f(archivo, ios::binary);
    if (!f) throw runtime_error("No se pudo abrir archivo de etiquetas");

    int32_t magic, num;
    f.read((char*)&magic, 4);
    f.read((char*)&num, 4);

    magic = __builtin_bswap32(magic);
    num = __builtin_bswap32(num);

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

void mostrarImagenConsola(const vector<float>& imagen) {
    const string escala = " .:-=+*#%@";  // 10 niveles de gris
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            float pixel = imagen[i * 28 + j];
            int nivel = static_cast<int>(pixel * (escala.size() - 1));
            cout << escala[nivel];
        }
        cout << '\n';
    }
}

float calcularLoss(const vector<float>& salida, const vector<float>& objetivo) {
    float loss = 0.0f;
    for (int i = 0; i < salida.size(); ++i) {
        loss += 0.5f * pow(salida[i] - objetivo[i], 2);  // MSE
    }
    return loss;
}

void entrenarModelo(PerceptronMulticapa& red, const vector<vector<float>>& imagenes,
                    const vector<vector<float>>& etiquetas, int epocas) {
    int cantidad = imagenes.size();

    for (int epoca = 0; epoca < epocas; ++epoca) {
        float total_loss = 0.0f;
        int aciertos = 0;

        for (int i = 0; i < cantidad; ++i) {
            red.establecerEntrada(imagenes[i]);
            auto salida = red.ejecutarRed();
            total_loss += calcularLoss(salida, etiquetas[i]);

            int pred = max_element(salida.begin(), salida.end()) - salida.begin();
            int real = max_element(etiquetas[i].begin(), etiquetas[i].end()) - etiquetas[i].begin();
            if (pred == real) ++aciertos;

            red.backpropagation(etiquetas[i]);
        }

        cout << "Época " << epoca + 1
             << " | Accuracy: " << (aciertos * 100.0 / cantidad)
             << "% | Loss: " << total_loss / cantidad << '\n';
    }
}

void probarModelo(PerceptronMulticapa& red, const vector<vector<float>>& imagenes_test,
                  const vector<vector<float>>& etiquetas_test) {
    int correctos = 0;

    for (int i = 0; i < imagenes_test.size(); ++i) {
        red.establecerEntrada(imagenes_test[i]);
        vector<float> salida = red.ejecutarRed();

        int prediccion = max_element(salida.begin(), salida.end()) - salida.begin();
        int real = max_element(etiquetas_test[i].begin(), etiquetas_test[i].end()) - etiquetas_test[i].begin();

        cout << "Imagen #" << i + 1 << "\n";
        mostrarImagenConsola(imagenes_test[i]);
        cout << "Etiqueta real: " << real << " | Predicción: " << prediccion << "\n\n";

        if (prediccion == real) ++correctos;
    }

    cout << "Precisión total en test: " << (correctos * 100.0 / imagenes_test.size()) << "%\n";
}


#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

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
    
    // Configuración de funciones de activación por capa
    vector<FuncionActivacion> funciones = {RELU, RELU, SOFTMAX}; // Capas ocultas: ReLU, Salida: Softmax
    
    int epocas = 10;

    if (argc > 1) {
        modelo = argv[1];
        if (argc > 2) {
            estructuraRed = parseEstructuraRed(argv[2]);
        }
        if (argc > 3) {
            epocas = stoi(argv[3]);
        }
    }

    PerceptronMulticapa red;

    if (filesystem::exists(modelo)) {
        cout << "Cargando modelo existente desde: " << modelo << endl;
        red.crearRed(estructuraRed, funciones);
        red.cargarPesos(modelo);
    } else {
        cout << "Archivo no encontrado. Entrenando nuevo modelo y guardando en: " << modelo << endl;
        red.crearRed(estructuraRed, funciones);

        const int cantidad = 1000;
        auto imagenes = leerImagenesMNIST("dataset/mnist/train-images.idx3-ubyte", cantidad);
        auto etiquetas = leerEtiquetasMNIST("dataset/mnist/train-labels.idx1-ubyte", cantidad);

        entrenarModelo(red, imagenes, etiquetas, epocas);

        red.guardarPesos(modelo);
    }

    // Prueba
    auto imagenes_test = leerImagenesMNIST("dataset/mnist/t10k-images.idx3-ubyte", 10);
    auto etiquetas_test = leerEtiquetasMNIST("dataset/mnist/t10k-labels.idx1-ubyte", 10);

    probarModelo(red, imagenes_test, etiquetas_test);

    return 0;
}