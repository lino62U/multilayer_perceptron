#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>  // Para leer el archivo

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42  // Semilla fija

using namespace std;

struct Neurona {
    float salida = 0.0f;
    float net = 0.0f;
    float delta = 0.0f;
    bool esBias = false;
    bool esSalida = false;

    vector<pair<Neurona*, float*>> entradas;
    vector<pair<Neurona*, float*>> salidas;

    float funcion_activacion(float x) {
        return 1.0f / (1.0f + exp(-x));  // Sigmoide
    }

    float derivada_activacion(float y) {
        return y * (1.0f - y);  // Derivada de la sigmoide
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
        salida = funcion_activacion(suma);
    }

    void calcularDeltaSalida(float objetivo) {
        delta = (salida - objetivo) * derivada_activacion(salida);
    }

    void calcularDeltaOculta() {
        float suma = 0.0f;
        for (auto& s : salidas) {
            suma += *(s.second) * s.first->delta;
        }
        delta = derivada_activacion(salida) * suma;
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

public:
    void crearRed(const vector<int>& estructura) {
        srand(RANDOM_SEED);  // Semilla fija

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
                capa.push_back(n);
            }

            capas.push_back(capa);
        }

        // Conectar neuronas capa a capa
        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                if (neurona->esBias) continue;  // El bias no recibe entradas
                for (auto& anterior : capas[i - 1]) {
                    float* peso = new float(((rand() % 100) / 100.0f) - 0.5f); // Peso entre -0.5 y 0.5
                    neurona->entradas.push_back({anterior, peso});
                    anterior->salidas.push_back({neurona, peso});
                }
            }
        }
    }

    void establecerEntrada(const vector<float>& entradas) {
        for (int i = 0; i < entradas.size(); ++i) {
            capas[0][i]->salida = entradas[i];
        }
    }

    float ejecutarRed() {
        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                neurona->calcularSalida();
            }
        }
        return capas.back()[0]->salida;
    }

    void backpropagation(const vector<float>& objetivos) {
        for (int i = 0; i < capas.back().size(); ++i) {
            capas.back()[i]->calcularDeltaSalida(objetivos[i]);
        }

        for (int i = capas.size() - 2; i > 0; --i) {
            for (auto& neurona : capas[i]) {
                if (!neurona->esBias)
                    neurona->calcularDeltaOculta();
            }
        }

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
            cout << "Capa " << i << ":\n";
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
};

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

int main() {
    PerceptronMulticapa red;
    red.crearRed({1, 1}); // 1 entrada (x), 1 salida (y)

    vector<vector<float>> datos = leerDatos("datos_sinteticos.txt"); // Leer los datos desde el archivo

    // Entrenamiento de la red
    for (int epoca = 0; epoca < 50; ++epoca) {
        for (int i = 0; i < datos.size(); ++i) {
            red.entrenar({datos[i][0]}, {datos[i][1]}, 1);  // Entrenamiento con una entrada
        }
    }

    // Imprimir los resultados en formato adecuado para Python
    ofstream archivoSalida("resultados.txt");
    for (int i = 0; i < datos.size(); ++i) {
        red.establecerEntrada({datos[i][0]});
        float salida = red.ejecutarRed();
        archivoSalida << datos[i][0] << "," << salida << endl;
    }

    archivoSalida.close();
    cout << "Resultados guardados en 'resultados.txt'." << endl;

    // Guardar predicciones para graficar en Python
    ofstream salidaPred("predicciones.csv");
    for (float x = 0.0f; x <= 1.0f; x += 0.01f) {
        red.establecerEntrada({x});
        float y = red.ejecutarRed();
        salidaPred << x << "," << y << endl;
    }
    salidaPred.close();

    return 0;
}
