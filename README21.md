
# **Documentación: Implementación de un Perceptrón Simple en C++**

Este programa en C++ implementa un perceptrón de una sola capa para modelar puertas lógicas básicas como **AND** y **OR**. El perceptrón es uno de los modelos más fundamentales en redes neuronales y aprende a clasificar entradas ajustando sus pesos durante el entrenamiento.

## **Objetivo**

Demostrar cómo un perceptrón puede aprender el comportamiento de puertas lógicas mediante entrenamiento supervisado.

## **Componentes Principales**

- **`Neuron` (estructura):** Representa una neurona individual con su salida, si es de sesgo (bias), y conexiones con pesos a otras neuronas.
- **`Layers` (clase):** Administra la arquitectura del perceptrón, creando y conectando capas de neuronas.
- **`SimplePerceptron` (clase):** Implementa el funcionamiento del perceptrón: propagación hacia adelante, retropropagación y entrenamiento.
- **`main` (función):** Crea y entrena perceptrones para las puertas AND y OR, mostrando los resultados.

## **Entrenamiento**

- **Aprendizaje supervisado** con una tasa de aprendizaje fija (`LEARNING_RATE = 0.1f`).
- Se utiliza una **función de activación escalón**.
- Los pesos se inicializan en `0.0`, aunque existe una semilla fija (`RANDOM_SEED = 42`) para reproducibilidad.

## **Estructura del Código**

### **Constantes**

- `LEARNING_RATE`: Tasa de aprendizaje (0.1). Controla cuánto se ajustan los pesos.
- `RANDOM_SEED`: Semilla fija para reproducibilidad (aunque los pesos actualmente se inician en 0.0).

### **Estructura `Neuron`**

Representa una neurona individual.

**Miembros:**

- `float output`: Valor de salida.
- `bool isBias`: Indica si es una neurona de sesgo (bias).
- `vector<pair<Neuron*, float*>> weights`: Conexiones a neuronas de la capa anterior, con sus respectivos pesos.

**Propósito:**  
Modela la operación de una neurona: suma ponderada, función de activación y opcionalmente actúa como bias.

### **Clase `Layers`**

Administra la estructura del perceptrón en capas.

**Miembros Privados:**

- `vector<vector<Neuron*>> layers`: Contenedor de capas (cada una es un vector de punteros a neuronas).

**Métodos:**

- `Layers()`: Constructor.
- `~Layers()`: Destructor, libera memoria para evitar fugas.
- `void buildNetwork(const vector<int>& estructura)`: Crea las capas, agrega neuronas bias excepto en la capa de salida, y conecta neuronas entre capas consecutivas con pesos iniciales en 0.0.
- `const vector<vector<Neuron*>>& getLayers() const`: Acceso de solo lectura a las capas.
- `vector<vector<Neuron*>>& getLayers()`: Acceso modificable.

### **Clase `SimplePerceptron`**

Contiene toda la lógica del perceptrón.

**Miembros Privados:**

- `Layers perceptron`: Instancia de la clase `Layers`.

**Métodos:**

- `SimplePerceptron(const vector<int>& estructura)`: Inicializa la red.
- `float activationFunction(float x)`: Función escalón. Devuelve 1 si `x >= 0`, si no, 0.
- `float forward()`: Realiza la propagación hacia adelante. Calcula las salidas de todas las neuronas.
- `void backpropagation(const vector<float>& targets)`: Calcula el error de cada salida y ajusta los pesos con:  
  `peso += LEARNING_RATE * error * entrada`.
- `void train(const vector<float>& input, const vector<float>& target, int epochs = 1000)`: Entrena el modelo con los pares entrada/objetivo.
- `void setInput(const vector<float>& inputs)`: Establece las entradas a la capa de entrada.
- `void printNetwork()`: Imprime el estado de la red (para depuración).
- `void printFinalWeights()`: Imprime los pesos finales aprendidos por la red.

### **Función `main`**

- Crea dos perceptrones: uno para **AND** y otro para **OR**, con estructura `{2, 1}`.
- Define entradas: `{(0,0), (0,1), (1,0), (1,1)}`.
- Define objetivos:
  - **AND:** `{0, 0, 0, 1}`
  - **OR:** `{0, 1, 1, 1}`
- Muestra los pesos iniciales.
- Entrena durante 5000 épocas.
- Imprime las tablas de verdad.
- Muestra los pesos finales.

## **Cómo Funciona el Perceptrón**

### **Arquitectura**

- Una capa de entrada (2 neuronas), una neurona de **bias**, y una neurona de salida.

### **Proceso de Aprendizaje**

1. Se ingresan los datos.
2. La neurona de salida calcula la suma ponderada de las entradas + bias.
3. Se aplica la función escalón:
   
   $\text{f}(x) = \begin{cases}
   1 & \text{si } x \geq 0 \\
   0 & \text{si } x < 0
   \end{cases}$

4. Se calcula el error:
   
   $\text{error} = \text{objetivo} - \text{salida}$

5. Se actualizan los pesos:
   
   $\text{peso}_{i} = \text{peso}_{i} + \eta \cdot \text{error} \cdot \text{entrada}_{i}$

6. Se repite durante múltiples épocas hasta que el error converge.

## **Tablas de Verdad y Pesos Finales**

### **Puerta AND**

| A | B | Salida |
| - | - | ------ |
| 0 | 0 | 0      |
| 0 | 1 | 0      |
| 1 | 0 | 0      |
| 1 | 1 | 1      |

**Pesos Finales (ejemplo):**

Capa 1 - Neurona de salida:  
Peso desde A:     0.4  
Peso desde B:     0.4  
Peso desde bias: -0.6

### **Puerta OR**

| A | B | Salida |
| - | - | ------ |
| 0 | 0 | 0      |
| 0 | 1 | 1      |
| 1 | 0 | 1      |
| 1 | 1 | 1      |

**Pesos Finales (ejemplo):**

Capa 1 - Neurona de salida:  
Peso desde A:     0.5  
Peso desde B:     0.5  
Peso desde bias: -0.2

## **Comparación con una Librería Externa**

Con el objetivo de validar el funcionamiento del perceptrón implementado manualmente en C++, se ha realizado una comparación con una implementación provista por una librería externa de aprendizaje automático. Para ello, se ha utilizado la biblioteca `scikit-learn` en el lenguaje Python, la cual ofrece una clase `Perceptron` de fácil configuración.

### **Configuración del Experimento**

Se ha replicado el mismo problema de clasificación de compuertas lógicas **AND** y **OR**, entrenando un modelo `Perceptron` en Python con los siguientes parámetros:

```python
from sklearn.linear_model import Perceptron

# Datos para compuerta lógica AND
X = [[0,0], [0,1], [1,0], [1,1]]
y_and = [0, 0, 0, 1]

# Modelo para AND
clf_and = Perceptron(max_iter=5000, eta0=0.1, random_state=42)
clf_and.fit(X, y_and)
print(clf_and.predict(X))  # Resultado esperado: [0 0 0 1]

# Datos para compuerta lógica OR
y_or = [0, 1, 1, 1]

# Modelo para OR
clf_or = Perceptron(max_iter=5000, eta0=0.1, random_state=42)
clf_or.fit(X, y_or)
print(clf_or.predict(X))  # Resultado esperado: [0 1 1 1]
```

### **Resultados Comparativos**

| Modelo               | Compuerta AND | Compuerta OR |
|----------------------|----------------|---------------|
| Implementación en C++ |     ✅ Correcto    |   ✅ Correcto   |
| `scikit-learn` (Python) | ✅ Correcto    |   ✅ Correcto   |

### **Observaciones**

- La biblioteca `scikit-learn` abstrae internamente el manejo de pesos, tasa de aprendizaje y función de activación, lo cual facilita su uso, pero oculta los detalles del proceso de entrenamiento.
- La implementación manual en C++ permite visualizar cada paso del cálculo, lo cual resulta ideal para fines educativos o de investigación en redes neuronales.
- En ambos casos, dado que las funciones **AND** y **OR** son linealmente separables, los perceptrones fueron capaces de aprenderlas sin dificultad.
