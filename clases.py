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

    @staticmethod
    def _clave_orden(ds):
        # Orden anatómico: posición en Z; si no existe, número de instancia
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        return int(ds.get("InstanceNumber", 0))

    def cargar_dicom(self):
        self.cortes = []  # evita duplicados si se llama dos veces

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

        # Dejar solo los cortes con el mismo tamaño que el primero
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

        # Ordenar de forma anatómica
        self.cortes.sort(key=self._clave_orden)

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

        # Matriz con TODOS los cortes ya ordenados
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