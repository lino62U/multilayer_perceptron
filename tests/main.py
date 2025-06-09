import pandas as pd
import matplotlib.pyplot as plt

def graficar_entrenamientos(lista_csv, etiquetas=None):
    if etiquetas is None:
        etiquetas = [f'Modelo {i+1}' for i in range(len(lista_csv))]

    colores = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown']
    marcadores = ['o', '^', 'x', 's', 'D', '*']  # círculo, triángulo, cruz, cuadrado, diamante, estrella

    # Primer gráfico: Loss
    plt.figure(figsize=(8, 5))
    for i, csv_file in enumerate(lista_csv):
        data = pd.read_csv(csv_file, sep='\t')
        data.columns = data.columns.str.strip()

        plt.plot(data['Epoch'], data['Loss'],
                 color=colores[i % len(colores)],
                 linestyle='-', marker=marcadores[i % len(marcadores)],
                 label=f'Loss - {etiquetas[i]}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
   # plt.title('Comparación de Optimización: Loss por Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Segundo gráfico: Accuracy
    plt.figure(figsize=(8, 5))
    for i, csv_file in enumerate(lista_csv):
        data = pd.read_csv(csv_file, sep='\t')
        data.columns = data.columns.str.strip()

        plt.plot(data['Epoch'], data['Accuracy(%)'],
                 color=colores[i % len(colores)],
                 linestyle='--', marker=marcadores[i % len(marcadores)],
                 label=f'Accuracy - {etiquetas[i]}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    #plt.title('Comparación de Optimización: Accuracy por Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ejemplo de uso:
csv_files = [
    'mnist_train_20epochs_adam.csv',
    'mnist_train_20epochs_sgd.csv',
    'mnist_train_20epochs_RMS.csv'
]
etiquetas = ['Adam', 'SGD', 'RMSProp']
graficar_entrenamientos(csv_files, etiquetas)
