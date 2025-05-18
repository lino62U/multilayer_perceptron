#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

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
        return 1.0f / (1.0f + exp(-x));
    }

    float derivada_activacion(float y) {
        return y * (1.0f - y);
    }
    
    float relu(float x) {
        return x > 0 ? x : 0.0f;
    }
    
    float relu_derivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
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
        //for (int i = 0; i < epocas; ++i) {
            establecerEntrada(entrada);
            ejecutarRed();
            backpropagation(objetivo);
        //}
    }

    void imprimirRed() {
        for (int i = 0; i < capas.size(); ++i) {
            cout << "Capa " << i << ":\n";
            for (auto& neurona : capas[i]) {
                cout << "  Salida: " << neurona->salida << ", Delta: " << neurona->delta;
                if (neurona->esBias) cout << " [bias]";
                cout << endl;
            }
        }
    }
    void imprimirPesosIniciales(const string& nombrePuerta) {
    cout << "\n==============================\n";
    cout << " Initial Weights for " << nombrePuerta << " Gate\n";
    cout << "==============================\n";

    for (int i = 0; i < capas.size(); ++i) { // Saltamos la capa de entrada
        cout << "Layer " << i << ":\n";
        for (int j = 0; j < capas[i].size(); ++j) {
            if (capas[i][j]->esBias) continue; // No imprimimos bias como destino
            cout << "  Neuron " << j << ":\n";
            int entradaIdx = 0;
            for (auto& entrada : capas[i][j]->entradas) {
                string origen;
                if (entrada.first->esBias) {
                    origen = "bias";
                } else {
                    origen = (entradaIdx == 0 ? "A" : "B"); // Asumimos 2 entradas
                    ++entradaIdx;
                }
                cout << "    From " << origen << ": weight = " << *(entrada.second) << endl;
            }
        }
    }
}

};

int main() {
    PerceptronMulticapa red;
    red.crearRed({2, 2, 1}); // Clasificador lineal

    red.imprimirPesosIniciales("AND");

    vector<vector<float>> entradas = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    vector<vector<float>> objetivos = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    for (int epoca = 0; epoca < 15; ++epoca) {
        for (int i = 0; i < entradas.size(); ++i) {
            red.entrenar(entradas[i], objetivos[i], 100);
        }
    }

    cout << "\nPesos aprendidos por el clasificador lineal:\n";
    red.imprimirRed();
    red.imprimirPesosIniciales("AND");

    cout << "\nResultados del clasificador:\n";
    for (int i = 0; i < entradas.size(); ++i) {
        red.establecerEntrada(entradas[i]);
        float salida = red.ejecutarRed();
        cout << entradas[i][0] << " XOR " << entradas[i][1] << " = " << salida << endl;
    }

    return 0;
}
