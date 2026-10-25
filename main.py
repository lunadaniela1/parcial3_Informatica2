import os
from clases import GestorEstudios


def menu():
    print("\n--- Análisis DICOM ---")
    print("1. Cargar un estudio")
    print("2. Rotar imagen (corregir ángulo)")
    print("3. Ecualizar un corte")
    print("4. Buscar y analizar huecos")
    print("5. Salir")
    return input("Ingresa una opción: ")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ruta_datos = os.path.join(base, "datos")
    ruta_resultados = os.path.join(base, "resultados")

    gestor = GestorEstudios(ruta_base=ruta_datos)

    os.makedirs(ruta_resultados, exist_ok=True)

    if not gestor.estudios_disponibles:
        print("Aviso: no hay carpetas en 'datos'.")

    while True:
        opc = menu()

        if opc == '1':
            print("Carpetas encontradas:", list(gestor.estudios_disponibles.keys()))
            nombre = input("¿Cuál estudio quieres abrir?: ")
            if gestor.seleccionar_estudio(nombre):
                print(f"Estudio '{gestor.nombre_actual}' cargado y volumen listo.")
            else:
                print("No se encontró o no se pudo cargar ese estudio.")

        elif opc == '2':
            if gestor.estudio_actual:
                ang = input("Grados de inclinación: ")
                gestor.estudio_actual.corregirAngulo(ang)
            else:
                print("Primero carga un estudio.")

        elif opc == '3':
            if gestor.estudio_actual:
                corte = input("Número del corte a procesar: ")
                gestor.estudio_actual.normalizarImagen(
                    corte, ruta_resultados, prefijo=gestor.nombre_actual
                )
            else:
                print("Primero carga un estudio.")

        elif opc == '4':
            if gestor.estudio_actual:
                if gestor.estudio_actual.segmentarHuecos(
                    ruta_resultados, prefijo=gestor.nombre_actual
                ):
                    print("Todo guardado en la carpeta resultados.")
            else:
                print("Primero carga un estudio.")

        elif opc == '5':
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()