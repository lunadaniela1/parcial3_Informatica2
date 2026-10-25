import os
import cv2
from clases import GestorEstudios

def main():
    ruta_datos = "datos"
    carpeta_resultados = "resultados"
    if not os.path.exists(carpeta_resultados):
        os.makedirs(carpeta_resultados)

    print("--- Sistema de Análisis DICOM ---")
    gestor = GestorEstudios(ruta_base=ruta_datos)
    
    if not gestor.estudios_disponibles:
        print(f"No se encontraron carpetas en '{ruta_datos}'.")
        return

    nombre_estudio = list(gestor.estudios_disponibles.keys())[0]
    gestor.seleccionar_estudio(nombre_estudio)
    estudio = gestor.estudio_actual
    
    corte_img = estudio.obtener_plano(eje=2)
    img_norm, img_binaria = estudio.preprocesar_imagen(corte_img)

    num_labels, labels, stats, centroids = estudio.identificar_componentes(img_binaria)
    id_roi = estudio.identificar_huecos_anatomicos(num_labels, stats)
    
    if id_roi:
        area_mm2, esp_x, esp_y = estudio.calcular_area_fisica(stats, id_roi, plano_eje=2)
        img_anotada = estudio.dibujar_etiquetas(img_norm, stats, centroids, id_roi, area_mm2)
        recorte, ancho_mm, alto_mm = estudio.recortar_region(img_norm, stats, id_roi, esp_x, esp_y)

        cv2.imwrite(os.path.join(carpeta_resultados, f"{nombre_estudio}_anotada.png"), img_anotada)
        cv2.imwrite(os.path.join(carpeta_resultados, f"{nombre_estudio}_recorte.png"), recorte)
        print("Imágenes guardadas con éxito en la carpeta resultados.")

if __name__ == "__main__":
    main()