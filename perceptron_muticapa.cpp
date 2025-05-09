#include <iostream>
#include <vector>
#include <cmath> // Para exp()
#include <cstdlib> // Para rand()

#define learning_rate 0.1

using namespace std;

struct Neurona {
    float salida;  // salida de la neurona
    float net;      // suma de pesos
    float delta;
    bool flag;      //neurona para saber si es nodo final

    vector<pair<Neurona*, float>> vecinos;  // Vecinos junto con el peso de la conexión

    // Constructor para inicializar la salida
    Neurona() {
        salida = 0.0f;  // Valor inicial para la salida
    }

    // Imprimir la neurona con sus vecinos y los pesos
    void imprimir(const string& nombre) {
        cout << nombre << " => salida: " << salida << ", Vecinos: ";
        for (const auto& vecino : vecinos) {
            cout << "Neurona(" << vecino.first << ", Peso: " << vecino.second << ") ";
        }
        cout << endl;
    }

    // Función para agregar un vecino con su peso
    void agregarVecino(Neurona* vecino, float peso) {
        vecinos.push_back(make_pair(vecino, peso));
    }

    // Función sigmoide como activación
    float funcion_activacion(float x) {
        return 1.0 / (1.0 + exp(-x));
    }
    // Derivada de la sigmoide usando su salida ya calculada
    float derivada_activacion(float salida) {
        return salida * (1.0 - salida);
    }

    // Calcular la salida de la neurona (sumar salidas * pesos)  
    float calcularSalida() {
        float suma = 0.0f;
        for (auto& vecino : vecinos) {
            suma += vecino.second* vecino.first->salida;  // Sumar peso * salida del vecino
        }
        
        net = suma;

        salida = funcion_activacion(suma);  // Aplicar la activación
        return salida;
    }

    // Error total para backpropagation (esperado - salida)
    void cacularErrorTotal(float target, Neurona *salida_j){
        if(flag){
            // Supón que ya tienes:
            float delta_i = (salida - target) * derivada_activacion(net);  // ya calculado
            float gradiente = delta_i * salida_j->salida;  // derivada del error respecto a w_ij

            // Luego actualizas el peso:
            peso = peso - learning_rate * gradiente;


        }else{

        }

    }
};

class PerceptronMulticapa {
private:
    vector<vector<Neurona*>> capas;  // Vector de capas, cada capa tiene un vector de neuronas
    vector<int> neuronasPorCapa;  // Vector que almacena el número de neuronas por capa
    int nCapas;

public:
    // Constructor para inicializar
    PerceptronMulticapa() : nCapas(0) {}

    // Crear las capas y conexiones entre ellas
    void crearRed(const vector<int>& neuronasPorCapa) {
        this->neuronasPorCapa = neuronasPorCapa;
        this->nCapas = neuronasPorCapa.size();

        // Crear todas las capas según lo especificado
        for (int i = 0; i < nCapas; ++i) {
            vector<Neurona*> capa;
            for (int j = 0; j < neuronasPorCapa[i]; ++j) {
                capa.push_back(new Neurona());
            }
            capas.push_back(capa);
        }

        // Conectar las neuronas entre capas (salidas -> capa1, capa1 -> capa2, etc.)
        // for (int i = 0; i < capas.size() - 1; ++i) {
        //     for (auto& neuronaSiguiente : capas[i + 1]) {
        //         for (auto& neuronaAnterior : capas[i]) {
        //             // La neurona de la capa siguiente agrega como vecino a la neurona anterior
        //             float peso = (rand() % 100) / 100.0f; // Peso aleatorio
        //             neuronaSiguiente->agregarVecino(neuronaAnterior, peso);
        //         }
        //     }
        // }

        for (int i = 1; i < capas.size(); ++i) {
            for (int j = 0; j < capas[i].size(); ++j) {
                for (int k = 0; k < capas[i - 1].size(); ++k) {
                    float peso = (rand() % 100) / 100.0f; // Peso aleatorio
                    capas[i][j]->agregarVecino(capas[i - 1][k], peso);
                }
            }
        }

    }

    // Función para ingresar salidas a las neuronas de la capa de salida
    void ingresarsalidas(const vector<float>& salidas) {
        int totalCapas = capas.size();

        for (int i = 0; i < salidas.size(); ++i) {
            capas[0][i]->salida = salidas[i];
        }
    }

    // Función para calcular la salida final
    float calcularSalida() {
        // Propagar las salidas a través de las capas y calcular las salidas
        int totalCapas = capas.size();

        for (int capa = 1; capa < totalCapas; ++capa) {  // inicia desde la capa 1
            for (auto& neurona : capas[capa]) {
                neurona->calcularSalida();
            }
        }

        // Obtener la salida final de la última capa (capa de salida)
        return capas[nCapas - 1][0]->salida;
    }

    float backpropagation(){
        // Propagar las salidas a través de las capas y calcular las salidas
        int totalCapas = capas.size();

        for (int capa = totalCapas -1; capa >= 0; --capa) {  // inicia desde la capa 1
            for (auto& neurona : capas[capa]) {
                neurona->calcularSalida();
            }
        }

        // Obtener la salida final de la última capa (capa de salida)
        return capas[nCapas - 1][0]->salida;

    }

    // Imprimir la estructura de la red
    void imprimirRed() {
        for (int i = 0; i < capas.size(); ++i) {
            cout << "Capa " << i + 1 << " (Neuronas: " << capas[i].size() << "):" << endl;
            for (int j = 0; j < capas[i].size(); ++j) {
                capas[i][j]->imprimir("Neurona " + to_string(j + 1));
            }
        }
    }
};


int main() {
    vector<int> estructuraCapas;

    cout << "¿Deseas ingresar manualmente la estructura de la red neuronal? (s/n): ";
    string respuesta;
    getline(cin, respuesta);

    if (respuesta == "s" || respuesta == "S") {
        cout << "Ingresa el número de capas totales (incluye salida y salida): ";
        int nCapas;
        cin >> nCapas;

        if (nCapas < 2) {
            cout << "Se requieren al menos 2 capas (salida y salida). Se usará la configuración por defecto.\n";
            estructuraCapas = {2, 2, 1};  // Por defecto: salida, oculta, salida
        } else {
            for (int i = 0; i < nCapas; ++i) {
                int neuronas;
                if (i == 0) {
                    cout << "Número de neuronas en la capa de salida: ";
                } else if (i == nCapas - 1) {
                    cout << "Número de neuronas en la capa de salida (sug: 1): ";
                } else {
                    cout << "Número de neuronas en la capa oculta " << i << ": ";
                }
                cin >> neuronas;
                estructuraCapas.push_back(neuronas);
            }
        }
    } else {
        cout << "Usando configuración por defecto: 3 capas (2 neuronas por capa, 1 en salida).\n";
        estructuraCapas = {2, 2, 1};  // Por defecto
    }

    int nsalidas = estructuraCapas[0];

    // Crear y usar la red
    PerceptronMulticapa perceptron;
    perceptron.crearRed(estructuraCapas);

    // Ingresar valores de salida
    vector<float> salidas;
    cout << "Ingresa los valores de salida (" << nsalidas << " valores separados por espacio): ";
    for (int i = 0; i < nsalidas; ++i) {
        float valor;
        cin >> valor;
        salidas.push_back(valor);
    }

    perceptron.ingresarsalidas(salidas);

    // Calcular la salida
    float salida = perceptron.calcularSalida();
    cout << "Salida final: " << salida << endl;

    // Imprimir red
    perceptron.imprimirRed();

    return 0;
}
