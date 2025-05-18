#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define LEARNING_RATE 0.1f
#define RANDOM_SEED 42

using namespace std;

struct Neurona {
    float salida = 0.0f;
    float net = 0.0f;
    float delta = 0.0f;
    bool esBias = false;

    vector<pair<Neurona*, float*>> entradas;
    vector<pair<Neurona*, float*>> salidas;

    float activacion(float x) const {
        return 1.0f / (1.0f + exp(-x));
    }

    float derivada(float y) const {
        return y * (1.0f - y);
    }

    void calcularSalida() {
        if (esBias) {
            salida = 1.0f;
            return;
        }

        net = 0.0f;
        for (const auto& [neurona, peso] : entradas) {
            net += *peso * neurona->salida;
        }
        salida = activacion(net);
    }

    void calcularDelta(bool esSalida, float objetivo = 0.0f) {
        if (esSalida) {
            delta = (salida - objetivo) * derivada(salida);
        } else {
            float suma = 0.0f;
            for (const auto& [neurona, peso] : salidas) {
                suma += *peso * neurona->delta;
            }
            delta = derivada(salida) * suma;
        }
    }

    void actualizarPesos() {
        if (esBias) return;
        for (auto& [neurona, peso] : entradas) {
            *peso -= LEARNING_RATE * delta * neurona->salida;
        }
    }
};

class PerceptronMulticapa {
    vector<vector<Neurona*>> capas;

    void conectarCapas() {
        srand(RANDOM_SEED);
        for (size_t i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                if (neurona->esBias) continue;
                for (auto& anterior : capas[i-1]) {
                    float* peso = new float((rand()%100)/100.0f - 0.5f);
                    neurona->entradas.emplace_back(anterior, peso);
                    anterior->salidas.emplace_back(neurona, peso);
                }
            }
        }
    }

public:
    ~PerceptronMulticapa() {
        for (auto& capa : capas) {
            for (auto& neurona : capa) {
                for (auto& [_, peso] : neurona->entradas) {
                    delete peso;
                }
                delete neurona;
            }
        }
    }

    void crearRed(const vector<int>& estructura) {
        for (size_t i = 0; i < estructura.size(); ++i) {
            vector<Neurona*> capa;
            int n_neuronas = estructura[i] + (i != estructura.size()-1); // +1 para bias
            
            for (int j = 0; j < n_neuronas; ++j) {
                Neurona* n = new Neurona();
                if (j == n_neuronas-1 && i != estructura.size()-1) {
                    n->esBias = true;
                    n->salida = 1.0f;
                }
                capa.push_back(n);
            }
            capas.push_back(capa);
        }
        conectarCapas();
    }

    void establecerEntrada(const vector<float>& entradas) {
        for (size_t i = 0; i < entradas.size(); ++i) {
            capas[0][i]->salida = entradas[i];
        }
    }

    float ejecutarRed() {
        for (size_t i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                neurona->calcularSalida();
            }
        }
        return capas.back()[0]->salida;
    }

    void backpropagation(const vector<float>& objetivos) {
        // Capa de salida
        for (size_t i = 0; i < capas.back().size(); ++i) {
            capas.back()[i]->calcularDelta(true, objetivos[i]);
        }

        // Capas ocultas
        for (int i = capas.size()-2; i > 0; --i) {
            for (auto& neurona : capas[i]) {
                if (!neurona->esBias) {
                    neurona->calcularDelta(false);
                }
            }
        }

        // Actualizar pesos
        for (size_t i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                neurona->actualizarPesos();
            }
        }
    }

    void entrenar(const vector<float>& entrada, const vector<float>& objetivo) {
        establecerEntrada(entrada);
        ejecutarRed();
        backpropagation(objetivo);
    }

    void imprimirRed() const {
        for (size_t i = 0; i < capas.size(); ++i) {
            cout << "Capa " << i << ":\n";
            for (const auto& neurona : capas[i]) {
                cout << "  Salida: " << neurona->salida << ", Delta: " << neurona->delta;
                if (neurona->esBias) cout << " [bias]";
                cout << endl;
            }
        }
    }

    void imprimirPesos(const string& nombre) const {
        cout << "\n=== Pesos para " << nombre << " ===\n";
        for (size_t i = 1; i < capas.size(); ++i) {
            cout << "Capa " << i << ":\n";
            for (size_t j = 0; j < capas[i].size(); ++j) {
                if (capas[i][j]->esBias) continue;
                cout << "  Neurona " << j << ":\n";
                for (size_t k = 0; k < capas[i][j]->entradas.size(); ++k) {
                    string origen = capas[i][j]->entradas[k].first->esBias ? "bias" : 
                                  (k == 0 ? "A" : "B");
                    cout << "    Desde " << origen << ": " << *capas[i][j]->entradas[k].second << endl;
                }
            }
        }
    }
};

int main() {
    PerceptronMulticapa red;
    red.crearRed({2, 2, 1});

    red.imprimirPesos("XOR Inicial");

    vector<vector<float>> entradas = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<vector<float>> objetivos = {{0}, {1}, {1}, {0}};

    for (int epoca = 0; epoca < 14000; ++epoca) {
        for (size_t i = 0; i < entradas.size(); ++i) {
            red.entrenar(entradas[i], objetivos[i]);
        }
    }

    cout << "\nRed entrenada:\n";
    red.imprimirRed();
    red.imprimirPesos("XOR Final");

    cout << "\nResultados:\n";
    for (const auto& entrada : entradas) {
        red.establecerEntrada(entrada);
        cout << entrada[0] << " XOR " << entrada[1] << " = " << red.ejecutarRed() << endl;
    }

    return 0;
}