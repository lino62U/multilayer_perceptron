#!/bin/bash

PROG_NAME="mi_perceptron"
BUILD_DIR="build"
FORCE_REBUILD=0

function mostrar_ayuda() {
    echo "Uso: $0 [--rebuild] <modelo.bin> <dataset> [\"<estructura>\"] [\"<activaciones>\"] [epocas]"
    echo ""
    echo "Opciones:"
    echo "  --rebuild         Fuerza recompilación sin preguntar."
    echo ""
    echo "Parámetros:"
    echo "  <modelo.bin>      Archivo donde se guarda o carga el modelo."
    echo "  <dataset>         Dataset a usar (ej: xor, mnist, and, or)."
    echo "  <estructura>      Estructura de la red separada por comas (ej: \"784,128,64,10\")."
    echo "  <activaciones>    Funciones de activación por capa (ocultas + salida), separadas por comas."
    echo "                    Debe haber una función por cada conexión entre capas."
    echo "                    Ejemplo: \"relu,relu,softmax\" para 3 conexiones."
    echo "                    Funciones válidas: sigmoid, relu, tanh, softmax."
    echo "  [epocas]          Número de épocas para entrenamiento (por defecto 20 para mnist, 3000 para xor)."
    echo ""
    echo "Ejemplo de uso:"
    echo "  $0 --rebuild modelo.bin xor \"2,2,1\" \"sigmoid,sigmoid\" 5000"
    echo "  $0 modelo.bin mnist \"784,128,64,10\" \"relu,relu,softmax\" 20"
}

# Verificar si hay que forzar recompilación
if [ "$1" == "--rebuild" ]; then
    FORCE_REBUILD=1
    shift
fi

# Si no hay argumentos restantes, mostrar ayuda
if [ "$#" -eq 0 ]; then
    mostrar_ayuda
    exit 1
fi

# Crear carpeta build si no existe
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creando directorio $BUILD_DIR..."
    mkdir "$BUILD_DIR"
fi

# Compilar o recompilar según bandera
if [ -f "$BUILD_DIR/$PROG_NAME" ]; then
    if [ "$FORCE_REBUILD" -eq 1 ]; then
        echo "Forzando recompilación..."
        cd "$BUILD_DIR" || exit
        cmake ..
        make
        cd ..
    else
        read -p "Ya existe un build. ¿Quieres recompilar? (s/n): " respuesta
        if [[ "$respuesta" =~ ^[Ss]$ ]]; then
            echo "Recompilando..."
            cd "$BUILD_DIR" || exit
            cmake ..
            make
            cd ..
        else
            echo "Usando build existente."
        fi
    fi
else
    echo "Compilando por primera vez..."
    cd "$BUILD_DIR" || exit
    cmake ..
    make
    cd ..
fi

# Ejecutar el programa con los argumentos restantes
./$BUILD_DIR/$PROG_NAME "$@"
