import pandas as pd
import matplotlib.pyplot as plt
import os

def graficar_entrenamientos(lista_csv, etiquetas=None):
    if etiquetas is None:
        etiquetas = [f'Modelo {i+1}' for i in range(len(lista_csv))]

    colores = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:brown']
    marcadores = ['o', '^', 'x', 's', 'D', '*']  # círculo, triángulo, cruz, cuadrado, diamante, estrella

    # Crear carpeta para guardar gráficos
    os.makedirs("graphics", exist_ok=True)

    # === Gráficos individuales por optimizador ===
    for i, csv_file in enumerate(lista_csv):
        data = pd.read_csv(csv_file, sep='\t')
        data.columns = data.columns.str.strip()

        color = colores[i % len(colores)]
        marker_train = marcadores[i % len(marcadores)]
        marker_test = marcadores[(i + 1) % len(marcadores)]

        etiqueta = etiquetas[i].lower()

        # Accuracy: Train vs Test
        plt.figure(figsize=(8, 5))
        plt.plot(data['Epoch'], data['Train Accuracy(%)'],
                 color=color, linestyle='-', marker=marker_train,
                 label='Train Accuracy')
        plt.plot(data['Epoch'], data['Test Accuracy(%)'],
                 color=color, linestyle='--', marker=marker_test,
                 label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy - {etiquetas[i]}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"graphics/accuracy_{i}_{etiqueta}.png")
        plt.show()

        # Loss: Train vs Test
        plt.figure(figsize=(8, 5))
        plt.plot(data['Epoch'], data['Train Loss'],
                 color=color, linestyle='-', marker=marker_train,
                 label='Train Loss')
        plt.plot(data['Epoch'], data['Test Loss'],
                 color=color, linestyle='--', marker=marker_test,
                 label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss - {etiquetas[i]}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"graphics/loss_{i}_{etiqueta}.png")
        plt.show()

    # === Gráfico comparativo: Train Loss ===
    plt.figure(figsize=(8, 5))
    for i, csv_file in enumerate(lista_csv):
        data = pd.read_csv(csv_file, sep='\t')
        data.columns = data.columns.str.strip()
        plt.plot(data['Epoch'], data['Train Loss'],
                 color=colores[i % len(colores)],
                 linestyle='-', marker=marcadores[i % len(marcadores)],
                 label=f'Train Loss - {etiquetas[i]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss por Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphics/train_loss_comparativo.png")
    plt.show()

    # === Gráfico comparativo: Train Accuracy ===
    plt.figure(figsize=(8, 5))
    for i, csv_file in enumerate(lista_csv):
        data = pd.read_csv(csv_file, sep='\t')
        data.columns = data.columns.str.strip()
        plt.plot(data['Epoch'], data['Train Accuracy(%)'],
                 color=colores[i % len(colores)],
                 linestyle='--', marker=marcadores[i % len(marcadores)],
                 label=f'Train Accuracy - {etiquetas[i]}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy por Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphics/train_accuracy_comparativo.png")
    plt.show()

    # === Gráfico comparativo: Test Loss ===
    plt.figure(figsize=(8, 5))
    for i, csv_file in enumerate(lista_csv):
        data = pd.read_csv(csv_file, sep='\t')
        data.columns = data.columns.str.strip()
        plt.plot(data['Epoch'], data['Test Loss'],
                 color=colores[i % len(colores)],
                 linestyle='-', marker=marcadores[i % len(marcadores)],
                 label=f'Test Loss - {etiquetas[i]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss por Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphics/test_loss_comparativo.png")
    plt.show()

    # === Gráfico comparativo: Test Accuracy ===
    plt.figure(figsize=(8, 5))
    for i, csv_file in enumerate(lista_csv):
        data = pd.read_csv(csv_file, sep='\t')
        data.columns = data.columns.str.strip()
        plt.plot(data['Epoch'], data['Test Accuracy(%)'],
                 color=colores[i % len(colores)],
                 linestyle='--', marker=marcadores[i % len(marcadores)],
                 label=f'Test Accuracy - {etiquetas[i]}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy por Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphics/test_accuracy_comparativo.png")
    plt.show()

# === Ejemplo de uso ===
csv_files = [
    'mnist_train_20epochs_adam.csv',
    'mnist_train_20epochs_sgd.csv',
    'mnist_train_20epochs_RMS.csv'
]
etiquetas = ['Adam', 'SGD', 'RMSProp']
graficar_entrenamientos(csv_files, etiquetas)
