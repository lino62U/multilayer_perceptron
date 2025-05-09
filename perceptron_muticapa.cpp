#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define LEARNING_RATE 0.1f

using namespace std;

struct Neurona {
    float salida = 0.0f;
    float net = 0.0f;
    float delta = 0.0f;
    bool esSalida = false;

    vector<pair<Neurona*, float*>> entradas; // Neuronas anteriores
    vector<pair<Neurona*, float*>> salidas;  // Neuronas siguientes

    float funcion_activacion(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    float derivada_activacion(float y) {
        return y * (1.0f - y);
    }

    void calcularSalida() {
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
        for (auto& entrada : entradas) {
            *(entrada.second) -= LEARNING_RATE * delta * entrada.first->salida;
        }
    }
};

class PerceptronMulticapa {
    vector<vector<Neurona*>> capas;

public:
    void crearRed(const vector<int>& estructura) {
        srand(static_cast<unsigned>(time(0)));

        for (int i = 0; i < estructura.size(); ++i) {
            vector<Neurona*> capa;
            for (int j = 0; j < estructura[i]; ++j) {
                Neurona* n = new Neurona();
                n->esSalida = (i == estructura.size() - 1);
                capa.push_back(n);
            }
            capas.push_back(capa);
        }

        for (int i = 1; i < capas.size(); ++i) {
            for (auto& neurona : capas[i]) {
                for (auto& anterior : capas[i - 1]) {
                    float* peso = new float((rand() % 100) / 100.0f);
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
        // Salida
        for (int i = 0; i < capas.back().size(); ++i) {
            capas.back()[i]->calcularDeltaSalida(objetivos[i]);
        }

        // Capas ocultas
        for (int i = capas.size() - 2; i > 0; --i) {
            for (auto& neurona : capas[i]) {
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
            cout << "Capa " << i << ":\n";
            for (auto& neurona : capas[i]) {
                cout << "  Salida: " << neurona->salida << ", Delta: " << neurona->delta << endl;
            }
        }
    }
};

int main() {
    PerceptronMulticapa red;
    red.crearRed({2, 2, 1}); // 2 entradas, 2 ocultas, 1 salida

    vector<float> entrada = {1.0f, 0.0f};
    vector<float> objetivo = {1.0f};

    red.entrenar(entrada, objetivo, 5000);
    
    red.establecerEntrada(entrada);
    float resultado = red.ejecutarRed();

    cout << "Salida final: " << resultado << endl;
    red.imprimirRed();
    
    return 0;
}
