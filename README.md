
# Multilayer Perceptron

Este proyecto implementa una red neuronal multicapa en C++ con soporte para diferentes funciones de activación, entrenamiento y prueba en datasets como XOR, AND, OR y MNIST.

---

## ⚙️ Preparación inicial

Antes de ejecutar el script por primera vez, asegúrate de darle permisos de ejecución:

```bash
chmod +x run.sh
````

Luego ya puedes usar el script normalmente.

---

## 🚀 Cómo compilar y ejecutar

Usa el script `run.sh` para compilar automáticamente y ejecutar el programa.

```bash
./run.sh [--rebuild] <modelo.bin> <dataset> ["<estructura>"] ["<activaciones>"] [epocas]
```

### 🛠️ Opciones

* `--rebuild`
  Fuerza la recompilación del programa sin preguntar.

---

### 📥 Parámetros

* `<modelo.bin>`
  Ruta al archivo donde se guardará o cargará el modelo (binario).

* `<dataset>`
  Nombre del dataset a utilizar:

  * `xor`
  * `and`
  * `or`
  * `mnist`

* `<estructura>` (opcional)
  Arquitectura de la red como lista de números separados por comas.
  Ejemplo: `"2,3,1"` o `"784,128,64,10"`

* `<activaciones>` (opcional)
  Lista de funciones de activación por cada capa (menos la capa de entrada).
  Ejemplo: `"sigmoid,sigmoid"` o `"relu,relu,softmax"`

  Funciones válidas:

  * `sigmoid`
  * `relu`
  * `tanh`
  * `softmax`

* `[epocas]` (opcional)
  Número de épocas de entrenamiento.
  Por defecto:

  * `3000` para `xor`, `and`, `or`
  * `20` para `mnist`

---

### 📌 Ejemplos

Entrenar una red para XOR con 2 capas ocultas y función sigmoid:

```bash
./run.sh --rebuild modelo_xor.bin xor "2,2,1" "sigmoid,sigmoid" 5000
```

Entrenar en MNIST con arquitectura profunda y funciones ReLU + Softmax:

```bash
./run.sh modelo_mnist.bin mnist "784,128,64,10" "relu,relu,softmax" 20
```

Cargar modelo ya entrenado sin recompilar:

```bash
./run.sh modelo_xor.bin xor
```

---

## 🧠 Notas

* Si no pasas `--rebuild`, el script te preguntará si deseas recompilar si ya existe un build anterior.
* Los pesos del modelo se guardan automáticamente en el archivo `.bin` que indiques.
* Si se encuentra un modelo existente, se cargará antes de entrenar.

---

## 🛠 Requisitos

* Compilador C++17 o superior
* CMake ≥ 3.10

---

## 🔧 Compilación manual (opcional)

Si no usas `run.sh`, puedes compilar directamente:

```bash
mkdir -p build
cd build
cmake ..
make
```

Y luego ejecutar:

```bash
./mi_perceptron <modelo.bin> <dataset> ...
```

---

