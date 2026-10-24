
import os
import matplotlib.pyplot as plt
from clases import EstudioDICOM

# Crear el objeto y cargar las imágenes DICOM
estudio = EstudioDICOM("datos/T2")
estudio.cargar_dicom()

# Reconstruir el volumen 3D
estudio.reconstruir_volumen()

# Corregir el ángulo
estudio.corregirAngulo(30)

# Seleccionar el corte central
numero_corte = estudio.volumen.shape[0] // 2

imagen_original = estudio.volumen[numero_corte]
imagen_corregida = estudio.imagen_rotada

# Crear figura con dos imágenes
plt.figure(figsize=(10, 5))

# Mostrar imagen original
plt.subplot(1, 2, 1)
plt.imshow(imagen_original, cmap="gray")
plt.title("Imagen original")
plt.axis("off")

# Mostrar imagen rotada
plt.subplot(1, 2, 2)
plt.imshow(imagen_corregida, cmap="gray")
plt.title(f"Imagen rotada {estudio.angulo_aplicado} grados")
plt.axis("off")

plt.tight_layout()

# Crear carpeta para guardar resultados
os.makedirs("resultados", exist_ok=True)

# Guardar la figura
plt.savefig("resultados/T2_rotacion.png", dpi=150)

# Mostrar la figura
plt.show()

corte = int(input(f"Corte a ecualizar (0 a {estudio.volumen.shape[0] - 1}): "))
estudio.normalizarImagen(corte, prefijo="T2")
