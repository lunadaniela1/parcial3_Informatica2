import os
import pydicom
import numpy as np
import cv2


class EstudioDICOM:

    def __init__(self, ruta):
        self.ruta = ruta
        self.cortes = []
        self.nombre_paciente = None
        self.modalidad = None
        self.filas = None
        self.columnas = None
        self.pixel_spacing = None      # [fila, columna] en mm
        self.slice_thickness = None
        self.tamano_voxel = None       # (alto, ancho, grosor) en mm
        self.matriz_pixeles = None     # arreglo (cortes, filas, columnas)
        self.volumen = None
        self.volumen_corregido= None
        self.imagen_rotada =None
        self.angulo_aplicado=None
        

    @staticmethod
    def _posicion_en_eje(ds):
        # Distancia del corte a lo largo de la normal del plano de imagen
        orient = np.array(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        pos = np.array(ds.ImagePositionPatient, dtype=float)
        return float(np.dot(pos, normal))

    def cargar_dicom(self):
        self.cortes = []  # evita duplicados si se llama dos veces

        # 1. Leer los archivos
        for raiz, _, archivos in os.walk(self.ruta):
            for archivo in archivos:
                ruta_archivo = os.path.join(raiz, archivo)
                try:
                    ds = pydicom.dcmread(ruta_archivo)
                except (pydicom.errors.InvalidDicomError, OSError):
                    continue

                if not hasattr(ds, "PixelData"):
                    continue

                self.cortes.append(ds)

        if len(self.cortes) == 0:
            print("No se encontraron imágenes DICOM en:", self.ruta)
            return False

        # 2. Filtrar: dejar solo los cortes con el mismo tamaño que el primero
        filas_ref = int(self.cortes[0].Rows)
        columnas_ref = int(self.cortes[0].Columns)
        total = len(self.cortes)
        self.cortes = [
            c for c in self.cortes
            if int(c.Rows) == filas_ref and int(c.Columns) == columnas_ref
        ]
        if len(self.cortes) < total:
            print("Aviso: se descartaron", total - len(self.cortes),
                  "cortes de tamaño distinto.")

        # 3. Ordenar (SOLO ESTE sort)
        if all(hasattr(ds, "ImagePositionPatient") and
               hasattr(ds, "ImageOrientationPatient") for ds in self.cortes):
            self.cortes.sort(key=self._posicion_en_eje)
        else:
            self.cortes.sort(key=lambda ds: int(ds.get("InstanceNumber", 0)))

        # 4. Leer los datos del estudio
        primer_corte = self.cortes[0]

        self.nombre_paciente = str(primer_corte.get("PatientName", "No disponible"))
        self.modalidad = str(primer_corte.get("Modality", "No disponible"))
        self.filas = int(primer_corte.Rows)
        self.columnas = int(primer_corte.Columns)

        self.pixel_spacing = primer_corte.get("PixelSpacing", [1, 1])
        self.slice_thickness = float(primer_corte.get("SliceThickness", 1))

        self.tamano_voxel = (
            float(self.pixel_spacing[0]),   # alto
            float(self.pixel_spacing[1]),   # ancho
            self.slice_thickness            # grosor del corte
        )

        # 5. Matriz con TODOS los cortes ya ordenados
        self.matriz_pixeles = np.stack([c.pixel_array for c in self.cortes])

        print("Estudio cargado correctamente.")
        print("Paciente:", self.nombre_paciente)
        print("Modalidad:", self.modalidad)
        print("Filas:", self.filas)
        print("Columnas:", self.columnas)
        print("Tamaño del vóxel (mm):", self.tamano_voxel)
        print("Cantidad de cortes:", len(self.cortes))
        print("Forma de la matriz:", self.matriz_pixeles.shape)
        return True

    def reconstruir_volumen(self):

        if self.matriz_pixeles is None:
            print("Primero debes cargar las imágenes DICOM.")
            return

        imagenes = []

        for imagen in self.matriz_pixeles:

            alto, ancho = imagen.shape

            imagen_doble = cv2.resize(
                imagen,
                (ancho * 2, alto * 2),
                interpolation=cv2.INTER_LINEAR
            )

            imagenes.append(imagen_doble)

        self.volumen = np.stack(imagenes, axis=0)

        print("Volumen 3D reconstruido correctamente.")
        print("Dimensiones del volumen:", self.volumen.shape)

    def corregirAngulo(self, angulo):

        if self.volumen is None:
            print("Primero debes reconstruir el volumen.")
            return False

        try:
            angulo = float(angulo)
        except (TypeError, ValueError):
            print("El ángulo debe ser un valor numérico.")
            return False

        alto, ancho = self.volumen.shape[1], self.volumen.shape[2]

        # Centro = punto medio de la imagen, como (x, y)
        centro = (ancho / 2, alto / 2)

        # Ángulo positivo = giro antihorario en OpenCV
        matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

        cortes_rotados = []
        for corte in self.volumen:
            rotado = cv2.warpAffine(
                corte,
                matriz_rotacion,
                (ancho, alto),
                flags=cv2.INTER_LINEAR
            )
            cortes_rotados.append(rotado)

        self.volumen_corregido = np.stack(cortes_rotados, axis=0)
        self.angulo_aplicado = angulo

    # Corte central, útil para visualizar el resultado
        self.imagen_rotada = self.volumen_corregido[self.volumen_corregido.shape[0] // 2]

        print("Corrección de ángulo aplicada correctamente.")
        print("Ángulo aplicado:", angulo)
        return True
