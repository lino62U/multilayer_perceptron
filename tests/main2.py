import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_optimizer_name(filename):
    base = os.path.basename(filename)
    parts = base.replace(".csv", "").split("_")
    for opt in ["adam", "sgd", "rmsprop"]:
        if opt.lower() in parts:
            return opt.upper()
    return "OTRO"

def graficar_entrenamientos_decay(lista_csv):
    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    marcadores = {
        'ADAM': 'o',
        'SGD': 'x',
        'RMSPROP': 's',
        'OTRO': '^'
    }

    # Crear carpeta para weight decay
    output_folder = "matrixx_decay"
    os.makedirs(output_folder, exist_ok=True)

    optim_labels = [extract_optimizer_name(f) for f in lista_csv]

    # === Gráficos individuales por optimizador ===
    for i, csv_file in enumerate(lista_csv):
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        label = optim_labels[i]
        color = colores[i % len(colores)]
        marker_train = marcadores.get(label, '^')
        marker_test = 'D'  # Diferente para test

        # Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_accuracy'], linestyle='-', marker=marker_train,
                 color=color, label='Train Accuracy')
        plt.plot(df['epoch'], df['test_accuracy'], linestyle='--', marker=marker_test,
                 color=color, label='Test Accuracy')
        plt.title(f"Accuracy - {label} (Decay)")
        plt.xlabel("Época")
        plt.ylabel("Accuracy (%)")
        plt.xlim(0, 20)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/accuracy_{label.lower()}_decay.png")
        plt.show()

        # Loss
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], linestyle='-', marker=marker_train,
                 color=color, label='Train Loss')
        plt.plot(df['epoch'], df['test_loss'], linestyle='--', marker=marker_test,
                 color=color, label='Test Loss')
        plt.title(f"Loss - {label} (Decay)")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.xlim(0, 20)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/loss_{label.lower()}_decay.png")
        plt.show()

    # === Gráficos comparativos ===
    def plot_comparativo(y_col, title, ylabel, filename, line_style):
        plt.figure(figsize=(10, 6))
        for i, csv_file in enumerate(lista_csv):
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            label = optim_labels[i]
            color = colores[i % len(colores)]
            marker = marcadores.get(label, '^')
            plt.plot(df['epoch'], df[y_col], marker=marker, linestyle=line_style,
                     color=color, label=label)
        plt.title(title + " (Decay)")
        plt.xlabel("Época")
        plt.ylabel(ylabel)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{filename}_decay.png")
        plt.show()

    plot_comparativo('test_accuracy', "Comparación de Test Accuracy", "Test Accuracy (%)", "comparative_test_accuracy", '-')
    plot_comparativo('test_loss', "Comparación de Test Loss", "Test Loss", "comparative_test_loss", '-')
    plot_comparativo('train_accuracy', "Comparación de Train Accuracy", "Train Accuracy (%)", "comparative_train_accuracy", '--')
    plot_comparativo('train_loss', "Comparación de Train Loss", "Train Loss", "comparative_train_loss", '-')

# === Uso para Weight Decay ===
csv_files = [
    'mnist_adam_decay_log.csv',
    'mnist_sgd_decay_log.csv',
    'mnist_rmsprop_decay_log.csv'
]
graficar_entrenamientos_decay(csv_files)
