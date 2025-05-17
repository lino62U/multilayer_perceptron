cmake_minimum_required(VERSION 3.10)
project(MiPerceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Añade la carpeta include para que encuentre los headers
include_directories(include)

# Archivos fuente
set(SOURCES
    src/main.cpp
    src/perceptron.cpp
    src/layers.cpp
    src/neuron.cpp
)

# Ejecutable principal
add_executable(mi_perceptron ${SOURCES})

# Opcional: aquí podrías enlazar librerías si usas alguna (por ejemplo, pthread, math, etc)
