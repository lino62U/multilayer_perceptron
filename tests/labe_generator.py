import os
import struct
from PIL import Image

def convertir_png_a_idx(imagenes_dir, output_images_path, output_labels_path):
    archivos = sorted([f for f in os.listdir(imagenes_dir) if f.endswith('.png')])

    if not archivos:
        raise ValueError("No se encontraron imágenes PNG en la carpeta")

    imagenes = []
    etiquetas = []

    for archivo in archivos:
        etiqueta = int(archivo.split('_')[0])
        ruta_imagen = os.path.join(imagenes_dir, archivo)
        imagen = Image.open(ruta_imagen).convert('L')  # escala de grises

        if imagen.size != (28, 28):
            imagen = imagen.resize((28, 28))  # redimensiona si es necesario

        pixeles = list(imagen.getdata())
        imagenes.append(pixeles)
        etiquetas.append(etiqueta)

    n = len(imagenes)
    rows, cols = 28, 28

    # Escribir archivo de imágenes (idx3-ubyte)
    with open(output_images_path, 'wb') as f:
        f.write(struct.pack('>IIII', 0x00000803, n, rows, cols))
        for img in imagenes:
            f.write(bytearray(img))

    # Escribir archivo de etiquetas (idx1-ubyte)
    with open(output_labels_path, 'wb') as f:
        f.write(struct.pack('>II', 0x00000801, n))
        f.write(bytearray(etiquetas))

    print(f"✅ {n} imágenes y etiquetas convertidas correctamente.")

# Ejemplo de uso:
convertir_png_a_idx(
    imagenes_dir='../data/mnist45',
    output_images_path='train-images.idx3-ubyte',
    output_labels_path='train-labels.idx1-ubyte'
)

