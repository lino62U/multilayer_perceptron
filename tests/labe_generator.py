import os
import struct
import numpy as np
from PIL import Image, ImageOps

def images_and_labels_to_idx(image_folder, image_output, label_output):
    images = []
    labels = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # Obtener la etiqueta desde el nombre del archivo (primer carácter)
                label = int(filename[0])
            except ValueError:
                print(f"[WARN] No se pudo extraer etiqueta de: {filename}")
                continue

            path = os.path.join(image_folder, filename)

            # Abrir imagen en escala de grises
            img = Image.open(path).convert('L')

            # Redimensionar a 28x28
            img = img.resize((28, 28))

            # Invertir colores
            img = ImageOps.invert(img)

            # Convertir a array
            img_array = np.array(img, dtype=np.uint8)

            images.append(img_array)
            labels.append(label)

    images_np = np.stack(images)
    labels_np = np.array(labels, dtype=np.uint8)

    num_images = len(images_np)
    rows, cols = 28, 28

    # === Escribir archivo .idx3-ubyte (imágenes)
    with open(image_output, 'wb') as f:
        f.write(struct.pack('>IIII', 2051, num_images, rows, cols))
        f.write(images_np.tobytes())

    # === Escribir archivo .idx1-ubyte (etiquetas)
    with open(label_output, 'wb') as f:
        f.write(struct.pack('>II', 2049, num_images))
        f.write(labels_np.tobytes())

    print(f"[INFO] Guardadas {num_images} imágenes en {image_output}")
    print(f"[INFO] Guardadas {num_images} etiquetas en {label_output}")

# === USO ===
images_and_labels_to_idx('../test_img', 'my-images.idx3-ubyte', 'my-labels.idx1-ubyte')
