import os
import matplotlib.pyplot as plt
from clases import EstudioDICOM

# Crear el objeto y cargar las imágenes DICOM (también reconstruye el volumen)
estudio = EstudioDICOM("datos/T2")
if not estudio.cargar_dicom():
    raise SystemExit("No se pudo cargar el estudio.")

# Corregir el ángulo
estudio.corregirAngulo(30)

# Seleccionar el corte central
numero_corte = estudio.volumen.shape[0] // 2

imagen_original = estudio.volumen[numero_corte]
imagen_corregida = estudio.imagen_rotada

# Crear figura con dos imágenes
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagen_original, cmap="gray")
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imagen_corregida, cmap="gray")
plt.title(f"Imagen rotada {estudio.angulo_aplicado} grados")
plt.axis("off")

plt.tight_layout()

# Guardar y mostrar
os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/T2_rotacion.png", dpi=150)
plt.show()

# Ecualización de un corte elegido por el usuario
# Ecualización de un corte elegido por el usuario
while True:
    try:
        corte = int(input(f"Corte a ecualizar (0 a {estudio.volumen.shape[0] - 1}): "))
        if estudio.normalizarImagen(corte, prefijo="T2"):
            break
    except ValueError:
        print("Debes escribir un número entero.")

# Selección de plano (el método pide plano y corte hasta que sean válidos)
if estudio.seleccionarPlano():

    plt.figure(figsize=(8, 6))
    plt.imshow(
        estudio.plano_seleccionado,
        cmap="gray",
        aspect=estudio.aspect_plano
    )
    plt.title(f"Plano {estudio.nombre_plano} - corte {estudio.indice_plano}")
    plt.axis("off")
    plt.savefig(
        f"resultados/T2_plano_{estudio.nombre_plano}_{estudio.indice_plano}.png",
        dpi=150
    )
    plt.show()

    if estudio.binarizarImagen():

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(estudio.plano_uint8, cmap="gray",
                   aspect=estudio.aspect_plano)
        plt.title("Plano original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(estudio.imagen_binaria, cmap="gray",
                   aspect=estudio.aspect_plano)
        plt.title(f"Imagen binarizada (umbral {estudio.umbral_usado})")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"resultados/T2_binarizacion_{estudio.nombre_plano}_{estudio.indice_plano}.png",
            dpi=150
        )
        plt.show()
                     # Aplicar apertura morfológica
        if estudio.aperturaMorfologica():

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(
                estudio.imagen_binaria,
                cmap="gray",
                aspect=estudio.aspect_plano
            )
            plt.title("Imagen binarizada")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(
                estudio.imagen_apertura,
                cmap="gray",
                aspect=estudio.aspect_plano
            )
            plt.title(f"Apertura (kernel {estudio.kernel_usado}, "
                      f"iter. {estudio.iteraciones_usadas})")
            plt.axis("off")

            plt.tight_layout()

            plt.savefig(
                f"resultados/T2_apertura_{estudio.nombre_plano}_{estudio.indice_plano}.png",
                dpi=150
            )

            plt.show()
