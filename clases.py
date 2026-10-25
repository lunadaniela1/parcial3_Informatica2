import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
        self.pixel_spacing_escalado = None
        self.tamano_voxel_escalado = None
        self.volumen_corregido = None
        self.imagen_rotada = None
        self.angulo_aplicado = None
        self.volumen_copia = None
        self.imagen_normalizada = None
        self.imagen_ecualizada = None
        self.corte_ecualizado = None
        self.plano_seleccionado = None
        self.nombre_plano = None
        self.indice_plano = None
        self.aspect_plano = None
        self.mm_por_pixel = None
        self.separacion_cortes = None
        self.plano_uint8 = None
        self.imagen_binaria = None
        self.umbral_usado = None
        self.modo_segmentacion = None
        self.imagen_apertura = None
        self.kernel_usado = None
        self.iteraciones_usadas = None
        self.area_hueco_mm2 = None
        self.imagen_anotada = None
        self.imagen_recorte = None


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
                except Exception:
                    # no es un DICOM (o está dañado): se ignora
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

        # Calcular la separación real entre cortes DICOM
        self.separacion_cortes = self.slice_thickness

        if len(self.cortes) > 1:
            if all(
                hasattr(ds, "ImagePositionPatient") and
                hasattr(ds, "ImageOrientationPatient")
                for ds in self.cortes
            ):
                posiciones = [
                    self._posicion_en_eje(ds)
                    for ds in self.cortes
                ]

                diferencias = np.abs(np.diff(posiciones))
                diferencias = diferencias[diferencias > 0]

                if len(diferencias) > 0:
                    self.separacion_cortes = float(np.median(diferencias))

        print("Separación entre cortes (mm):", self.separacion_cortes)

        self.tamano_voxel = (
            float(self.pixel_spacing[0]),   # alto
            float(self.pixel_spacing[1]),   # ancho
            self.slice_thickness            # grosor del corte
        )

        # 5. Matriz con TODOS los cortes ya ordenados
        try:
            self.matriz_pixeles = np.stack([c.pixel_array for c in self.cortes])
        except Exception as error:
            print("No se pudo leer la matriz de píxeles (¿DICOM comprimido?):", error)
            return False

        print("Estudio cargado correctamente.")
        print("Paciente:", self.nombre_paciente)
        print("Modalidad:", self.modalidad)
        print("Filas:", self.filas)
        print("Columnas:", self.columnas)
        print("Tamaño del vóxel (mm):", self.tamano_voxel)
        print("Cantidad de cortes:", len(self.cortes))
        print("Forma de la matriz:", self.matriz_pixeles.shape)

        # Escalado al doble y reconstrucción 3D automáticos (punto 3)
        return self.reconstruir_volumen()

    def reconstruir_volumen(self):

        if self.matriz_pixeles is None:
            print("Primero debes cargar las imágenes DICOM.")
            return False

        imagenes = []

        for imagen in self.matriz_pixeles:

            # cv2.resize no admite todos los tipos (por ejemplo int32)
            if imagen.dtype not in (np.uint8, np.uint16, np.int16,
                                    np.float32, np.float64):
                imagen = imagen.astype(np.float32)

            alto, ancho = imagen.shape

            # OpenCV recibe el tamaño como (ancho, alto)
            imagen_doble = cv2.resize(
                imagen,
                (ancho * 2, alto * 2),
                interpolation=cv2.INTER_LINEAR
            )

            imagenes.append(imagen_doble)

        self.volumen = np.stack(imagenes, axis=0)

        # Al duplicar filas y columnas, cada píxel representa la mitad de mm
        self.pixel_spacing_escalado = (
            float(self.pixel_spacing[0]) / 2,
            float(self.pixel_spacing[1]) / 2
        )
        self.tamano_voxel_escalado = (
            self.pixel_spacing_escalado[0],
            self.pixel_spacing_escalado[1],
            self.slice_thickness    # el eje de los cortes no se escala
        )

        print("Volumen 3D reconstruido correctamente.")
        print("Dimensiones del volumen:", self.volumen.shape)
        print("Tamaño del vóxel escalado (mm):", self.tamano_voxel_escalado)
        return True

    def corregirAngulo(self, angulo):

        if self.volumen is None:
            print("Primero debes reconstruir el volumen.")
            return False

        try:
            angulo = float(angulo)
        except (TypeError, ValueError):
            print("El ángulo debe ser un valor numérico.")
            return False
        
        if not np.isfinite(angulo):
            print("El ángulo debe ser un valor numérico finito.")
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
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(corte.min())   # las esquinas nuevas quedan como el fondo
            )
            cortes_rotados.append(rotado)

        self.volumen_corregido = np.stack(cortes_rotados, axis=0)
        self.angulo_aplicado = angulo

        # Corte central, útil para visualizar el resultado
        self.imagen_rotada = self.volumen_corregido[self.volumen_corregido.shape[0] // 2]

        print("Corrección de ángulo aplicada correctamente.")
        print("Ángulo aplicado:", angulo)
        return True

    def normalizarImagen(self, numero_corte, carpeta_salida="resultados", prefijo="estudio"):

        # Usar el volumen corregido si existe; si no, el volumen escalado
        base = self.volumen_corregido if self.volumen_corregido is not None else self.volumen

        if base is None:
            print("Primero debes cargar el estudio y reconstruir el volumen.")
            return False

        try:
            numero_corte = int(numero_corte)
        except (TypeError, ValueError):
            print("El número de corte debe ser un entero.")
            return False

        if not 0 <= numero_corte < base.shape[0]:
            print("Corte fuera de rango. Valores válidos: 0 a", base.shape[0] - 1)
            return False

        # Copia del volumen para no alterar el original
        self.volumen_copia = base.copy()
        imagen = self.volumen_copia[numero_corte]

        # Normalizar a uint8 (0-255) y ecualizar
        self.imagen_normalizada = cv2.normalize(
            imagen, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        self.imagen_ecualizada = cv2.equalizeHist(self.imagen_normalizada)
        self.corte_ecualizado = numero_corte

        # Graficar: imágenes arriba, histogramas abajo
        fig, ejes = plt.subplots(2, 2, figsize=(10, 8))

        ejes[0, 0].imshow(self.imagen_normalizada, cmap="gray")
        ejes[0, 0].set_title(f"Corte {numero_corte} normalizado (uint8)")
        ejes[0, 0].axis("off")

        ejes[0, 1].imshow(self.imagen_ecualizada, cmap="gray")
        ejes[0, 1].set_title("Imagen ecualizada")
        ejes[0, 1].axis("off")

        ejes[1, 0].hist(self.imagen_normalizada.ravel(), bins=256, range=(0, 255))
        ejes[1, 0].set_title("Histograma antes de ecualizar")

        ejes[1, 1].hist(self.imagen_ecualizada.ravel(), bins=256, range=(0, 255))
        ejes[1, 1].set_title("Histograma después de ecualizar")

        plt.tight_layout()

        # Guardar el gráfico y la imagen ecualizada
        os.makedirs(carpeta_salida, exist_ok=True)
        plt.savefig(os.path.join(carpeta_salida, f"{prefijo}_ecualizacion_grafico.png"), dpi=150)
        cv2.imwrite(os.path.join(carpeta_salida, f"{prefijo}_ecualizada.png"), self.imagen_ecualizada)

        plt.show()
        plt.close(fig)

        print("Imagen normalizada y ecualizada correctamente.")
        print("Tipo de dato:", self.imagen_ecualizada.dtype)
        return True


    def seleccionarPlano(self, plano=None, numero_corte=None):

        # Usar el volumen corregido si existe; si no, el escalado
        volumen = self.volumen_corregido if self.volumen_corregido is not None else self.volumen

        if volumen is None:
            print("Primero debes cargar el volumen.")
            return False

        # --- Pedir el plano hasta que sea válido ---
        planos_validos = ("transversal", "coronal", "sagital")

        if plano is not None:
            plano = str(plano).strip().lower()

        while plano not in planos_validos:
            if plano is not None:
                print("Plano no válido. Usa: transversal, coronal o sagital.")
            plano = input("Plano (transversal/coronal/sagital): ").strip().lower()

        # Eje del volumen que recorre los cortes de cada plano
        eje = {"transversal": 0, "coronal": 1, "sagital": 2}[plano]
        maximo = volumen.shape[eje]

        # --- Pedir el corte hasta que sea válido ---
        while True:
            if numero_corte is None:
                numero_corte = input(
                    f"Número de corte para el plano {plano} (0 a {maximo - 1}): "
                )

            try:
                valor = int(numero_corte)
            except (TypeError, ValueError):
                print("El número de corte debe ser un entero.")
                numero_corte = None
                continue

            if 0 <= valor < maximo:
                numero_corte = valor
                break

            print(f"Corte fuera de rango. Valores válidos: 0 a {maximo - 1}")
            numero_corte = None

        # Tamaño de píxel escalado (mm) y grosor del corte (mm)
        sp_fila, sp_col = self.pixel_spacing_escalado
        grosor = self.separacion_cortes

        if plano == "transversal":
            imagen = volumen[numero_corte, :, :]
            self.aspect_plano = sp_fila / sp_col
            self.mm_por_pixel = (sp_fila, sp_col)      # (alto, ancho)

        elif plano == "coronal":
            imagen = volumen[:, numero_corte, :]
            self.aspect_plano = grosor / sp_col
            self.mm_por_pixel = (grosor, sp_col)

        else:  # sagital
            imagen = volumen[:, :, numero_corte]
            self.aspect_plano = grosor / sp_fila
            self.mm_por_pixel = (grosor, sp_fila)

        self.plano_seleccionado = imagen.copy()
        self.nombre_plano = plano
        self.indice_plano = numero_corte

        print("Plano seleccionado:", plano, "- corte", numero_corte)
        print("Dimensiones:", imagen.shape)
        print("Aspect:", round(self.aspect_plano, 3))
        return True

    def binarizarImagen(self, umbral=None, segmentar=None):

        if self.plano_seleccionado is None:
            print("Primero debes seleccionar un plano.")
            return False
        if segmentar is not None:
               segmentar = str(segmentar).strip().lower()
        while segmentar not in ("claro", "oscuro"):
            if segmentar is not None:
                   print("Opción no válida. Escribe: claro u oscuro.")
            segmentar = input("¿Segmentar zonas claras u oscuras? (claro/oscuro): ").strip().lower()

        # Pedir el umbral hasta que sea válido (0 a 255)
        while True:
            if umbral is None:
                umbral = input("Umbral (0 a 255): ")

            try:
                valor = int(umbral)
            except (TypeError, ValueError):
                print("El umbral debe ser un número entero.")
                umbral = None
                continue

            if 0 <= valor <= 255:
                umbral = valor
                break

            print("Umbral fuera de rango. Valores válidos: 0 a 255")
            umbral = None

        # Normalizar el plano a uint8 (se reutiliza después para dibujar en RGB)
        self.plano_uint8 = cv2.normalize(
            self.plano_seleccionado,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        # INV: lo oscuro (huecos) queda en blanco
        # claro  -> BINARY:     lo que supera el umbral queda en blanco
        # oscuro -> BINARY_INV: lo que está por debajo del umbral queda en blanco
        tipo = cv2.THRESH_BINARY if segmentar == "claro" else cv2.THRESH_BINARY_INV
        _, self.imagen_binaria = cv2.threshold(self.plano_uint8, umbral, 255, tipo)
        self.umbral_usado = umbral
        self.modo_segmentacion = segmentar

        print("Imagen binarizada correctamente.")
        print("Umbral utilizado:", umbral)
        return True

    def aperturaMorfologica(self, tamano_kernel=None, iteraciones=None):

        if self.imagen_binaria is None:
            print("Primero debes binarizar la imagen.")
            return False

        # Pedir el tamaño del kernel hasta que sea válido
        while True:
            if tamano_kernel is None:
                tamano_kernel = input("Tamaño del kernel (entero mayor que 0): ")
            try:
                valor = int(tamano_kernel)
            except (TypeError, ValueError):
                print("El tamaño del kernel debe ser un número entero.")
                tamano_kernel = None
                continue
            if valor >= 1:
                tamano_kernel = valor
                break
            print("El tamaño del kernel debe ser mayor que cero.")
            tamano_kernel = None

        # Pedir las iteraciones hasta que sean válidas
        while True:
            if iteraciones is None:
                iteraciones = input("Iteraciones (entero mayor que 0): ")
            try:
                valor = int(iteraciones)
            except (TypeError, ValueError):
                print("Las iteraciones deben ser un número entero.")
                iteraciones = None
                continue
            if valor >= 1:
                iteraciones = valor
                break
            print("Las iteraciones deben ser mayores que cero.")
            iteraciones = None

        # Crear el kernel cuadrado
        kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)

        # Apertura = erosión seguida de dilatación
        self.imagen_apertura = cv2.morphologyEx(
            self.imagen_binaria,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iteraciones
        )

        self.kernel_usado = tamano_kernel
        self.iteraciones_usadas = iteraciones

        print("Apertura morfológica aplicada correctamente.")
        print("Tamaño del kernel:", tamano_kernel)
        print("Iteraciones:", iteraciones)

        if cv2.countNonZero(self.imagen_apertura) == 0:
            print("Aviso: la apertura eliminó todo. Prueba un kernel o "
                  "unas iteraciones más pequeñas.")
            
        return True
    
    def _reducir_apertura_si_vacia(self):
        # Si la apertura elimina todo, baja las iteraciones a 1 y luego reduce
        # el kernel a la mitad hasta que quede algo (con kernel 1 no borra nada).
        k, it = self.kernel_usado, self.iteraciones_usadas
        cambio = False
        while cv2.countNonZero(self.imagen_apertura) == 0:
            if cv2.countNonZero(self.imagen_binaria) == 0:
                return False
            if it > 1:
                it = 1
            elif k > 1:
                k = max(1, k // 2)
            else:
                return False
            kernel = np.ones((k, k), np.uint8)
            self.imagen_apertura = cv2.morphologyEx(
                self.imagen_binaria, cv2.MORPH_OPEN, kernel, iterations=it
            )
            cambio = True
        if cambio:
            self.kernel_usado = k
            self.iteraciones_usadas = it
            print(f"Aviso: la apertura borraba todo; se ajustó a kernel {k} e iteraciones {it}.")
        return True
    
        #commit 9: Identificar cada componente.
    def identificar_componentes(self, imagen_binaria):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, connectivity=8)
        return num_labels, labels, stats, centroids
    #commit 10: Area en milimetros
    def calcular_area_fisica(self, stats, index):
        # Usa los mm por píxel del plano elegido (ya incluyen el escalado x2
        # y el grosor/separación de los cortes), así el área es correcta en
        # los tres planos.
        area_pixeles = stats[index, cv2.CC_STAT_AREA]
        alto_mm, ancho_mm = self.mm_por_pixel

        area_mm2 = float(area_pixeles) * alto_mm * ancho_mm
        return area_mm2, ancho_mm, alto_mm   # (area, mm por px en x, mm por px en y)
    
    
    #commit 11: Solucion al ruido o al fondo, para imagenes limpias

    def identificar_huecos_anatomicos(self, num_labels, stats, min_area=50, max_area=None,
                                      permitir_borde=False):
        # Descarta el fondo (label 0) y todo componente que toque el borde
        # de la imagen (aire exterior); de los restantes devuelve el mayor.
        # Con permitir_borde=True se aceptan los que tocan el borde, salvo
        # los que ocupan más de la mitad de la imagen (el fondo).
        alto, ancho = self.imagen_apertura.shape
        area_total = alto * ancho
        solo_laterales = self.nombre_plano in ("coronal", "sagital")
        mejor_id = None
        mejor_area = 0

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            toca_lateral = x == 0 or (x + w) >= ancho
            toca_vertical = y == 0 or (y + h) >= alto
            toca_borde = toca_lateral or (toca_vertical and not solo_laterales)
            if permitir_borde:
                if area > 0.5 * area_total:
                    continue
            elif toca_borde:
                continue
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            if area > mejor_area:
                mejor_id = i
                mejor_area = area

        return mejor_id

    #commit 12: poner un indicador de lugar y texto de area

    def dibujar_etiquetas(self, imagen_original, stats, centroids, id_componente,
                          area_mm2, escribir_texto=True):
        # Pasa a color (BGR/RGB) para poder dibujar el círculo en color
        if len(imagen_original.shape) == 2:
            img_color = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)
        else:
            img_color = imagen_original.copy()

        cx, cy = int(centroids[id_componente][0]), int(centroids[id_componente][1])
        w = stats[id_componente, cv2.CC_STAT_WIDTH]
        h = stats[id_componente, cv2.CC_STAT_HEIGHT]
        radio = int(max(w, h) / 2) + 5

        cv2.circle(img_color, (cx, cy), radio, (0, 128, 255), 3)

        if escribir_texto:
            texto = f"{area_mm2:.1f} mm2" 
            escala = max(0.5, img_color.shape[1] / 1000)
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
            tx = min(max(0, cx - radio), max(0, img_color.shape[1] - tw - 5))
            ty = max(th + 5, cy - radio - 10)
            cv2.putText(img_color, texto, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 255, 0), 2)

        return img_color
    #commit 13: funcion de para recortar

    def recortar_region(self, imagen, stats, id_componente, espaciado_x, espaciado_y):
        x = stats[id_componente, cv2.CC_STAT_LEFT]
        y = stats[id_componente, cv2.CC_STAT_TOP]
        w = stats[id_componente, cv2.CC_STAT_WIDTH]
        h = stats[id_componente, cv2.CC_STAT_HEIGHT]
        
        # Le damos un pequeño borde para que el circulo no quede cortado
        borde = 40
        y1 = max(0, y - borde)
        y2 = min(imagen.shape[0], y + h + borde)
        x1 = max(0, x - borde)
        x2 = min(imagen.shape[1], x + w + borde)
        
        recorte = imagen[y1:y2, x1:x2]
        
        ancho_mm = w * espaciado_x
        alto_mm = h * espaciado_y
        
        return recorte, ancho_mm, alto_mm
    def graficar_resultado(self, img_anotada, img_apertura_color, recorte,
                           ancho_mm, alto_mm, area_mm2,
                           carpeta_salida="resultados", prefijo="estudio"):
        # Dimensiones físicas del corte completo
        alto_corte_mm = self.plano_uint8.shape[0] * self.mm_por_pixel[0]
        ancho_corte_mm = self.plano_uint8.shape[1] * self.mm_por_pixel[1]

        fig, ejes = plt.subplots(1, 3, figsize=(18, 6))

        ejes[0].imshow(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB),
                       aspect=self.aspect_plano)
        ejes[0].set_title(f"Corte {self.nombre_plano} #{self.indice_plano}\n"
                          f"Dimensiones: {ancho_corte_mm:.1f} x {alto_corte_mm:.1f} mm")

        ejes[1].imshow(cv2.cvtColor(img_apertura_color, cv2.COLOR_BGR2RGB),
                       aspect=self.aspect_plano)
        ejes[1].set_title(f"Binarizada ({self.modo_segmentacion}, umbral {self.umbral_usado}) + apertura\n"
                        f"kernel {self.kernel_usado}, iteraciones {self.iteraciones_usadas}")

        ejes[2].imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB),
                       aspect=self.aspect_plano)
        ejes[2].set_title(f"Región de interés\nAncho: {ancho_mm:.1f} mm | "
                          f"Alto: {alto_mm:.1f} mm | Área: {area_mm2:.1f} mm²")

        plt.tight_layout()

        os.makedirs(carpeta_salida, exist_ok=True)
        ruta = os.path.join(carpeta_salida, f"{prefijo}_{self.nombre_plano}_segmentacion.png")
        fig.savefig(ruta, dpi=150)

        plt.show()
        plt.close(fig)
        return ruta

    def segmentarHuecos(self, carpeta_salida="resultados", prefijo="estudio"):
        # Método del punto 6: plano -> binarizar -> apertura -> componentes
        # -> área en mm² -> círculo -> graficar y guardar.
        if not self.seleccionarPlano():
            return False
        if not self.binarizarImagen():
            return False
                # Respaldo 1: si el umbral no segmenta nada, se usa Otsu
        if cv2.countNonZero(self.imagen_binaria) == 0:
            tipo = cv2.THRESH_BINARY if self.modo_segmentacion == "claro" else cv2.THRESH_BINARY_INV
            umbral_otsu, self.imagen_binaria = cv2.threshold(
                self.plano_uint8, 0, 255, tipo + cv2.THRESH_OTSU
            )
            self.umbral_usado = int(umbral_otsu)
            print(f"Aviso: ese umbral no segmentó nada; se usó Otsu ({self.umbral_usado}).")

        if not self.aperturaMorfologica():
            return False

        # Respaldo 2: si la apertura borra todo, se reducen iteraciones y luego el kernel
        if not self._reducir_apertura_si_vacia():
            print("No hay nada que segmentar en este corte. Prueba otro corte o plano.")
            return False

        num_labels, labels, stats, centroids = self.identificar_componentes(self.imagen_apertura)

        # Respaldo 3: si no hay hueco, se relajan los criterios paso a paso
        id_hueco = None
        for min_area, permitir_borde, aviso in (
            (50, False, None),
            (1, False, "Aviso: se aceptaron huecos más pequeños (< 50 px)."),
            (1, True, "Aviso: no hay huecos cerrados; se aceptó una región que toca el borde."),
        ):
            id_hueco = self.identificar_huecos_anatomicos(
                num_labels, stats, min_area=min_area, permitir_borde=permitir_borde
            )
            if id_hueco is not None:
                if aviso:
                    print(aviso)
                break

        if id_hueco is None:
            print("No se encontró ningún hueco en este corte. "
                  "Prueba otro corte, umbral o plano.")
            return False

        area_mm2, esp_x, esp_y = self.calcular_area_fisica(stats, id_hueco)
        self.area_hueco_mm2 = area_mm2

        # Corte uint8 -> color, con círculo y texto del área
        self.imagen_anotada = self.dibujar_etiquetas(
            self.plano_uint8, stats, centroids, id_hueco, area_mm2
        )
        # Lo mismo sobre la imagen segmentada (como en el ejemplo del enunciado)
        apertura_anotada = self.dibujar_etiquetas(
            self.imagen_apertura, stats, centroids, id_hueco, area_mm2
        )
        # Recorte sin texto (el texto se muestra en el título de la gráfica)
        sin_texto = self.dibujar_etiquetas(
            self.plano_uint8, stats, centroids, id_hueco, area_mm2,
            escribir_texto=False
        )
        self.imagen_recorte, ancho_mm, alto_mm = self.recortar_region(
            sin_texto, stats, id_hueco, esp_x, esp_y
        )

        os.makedirs(carpeta_salida, exist_ok=True)
        cv2.imwrite(os.path.join(carpeta_salida, f"{prefijo}_{self.nombre_plano}_anotada.png"),
                    self.imagen_anotada)
        cv2.imwrite(os.path.join(carpeta_salida, f"{prefijo}_{self.nombre_plano}_recorte.png"),
                    self.imagen_recorte)

        self.graficar_resultado(self.imagen_anotada, apertura_anotada, self.imagen_recorte,
                                ancho_mm, alto_mm, area_mm2, carpeta_salida, prefijo)

        print(f"Área del hueco: {area_mm2:.1f} mm²")
        print(f"Dimensiones: {ancho_mm:.1f} x {alto_mm:.1f} mm")
        return True


class GestorEstudios:
    def __init__(self, ruta_base="datos"):
        self.ruta_base = ruta_base
        self.estudios_disponibles = self._escanear_directorios()
        self.estudios_cargados = {}     # nombre -> EstudioDICOM ya cargado
        self.estudio_actual = None
        self.nombre_actual = None

    def _escanear_directorios(self):
        estudios = {}
        if os.path.exists(self.ruta_base):
            for nombre in os.listdir(self.ruta_base):
                ruta_completa = os.path.join(self.ruta_base, nombre)
                if os.path.isdir(ruta_completa):
                    estudios[nombre] = ruta_completa
        return estudios

    def seleccionar_estudio(self, nombre_estudio):
        # Acepta el nombre sin importar mayúsculas/minúsculas
        nombre = next(
            (n for n in self.estudios_disponibles
             if n.lower() == str(nombre_estudio).strip().lower()),
            None
        )
        if nombre is None:
            return False

        # Si ya se cargó antes, se reutiliza el objeto
        if nombre in self.estudios_cargados:
            self.estudio_actual = self.estudios_cargados[nombre]
            self.nombre_actual = nombre
            return True

        estudio = EstudioDICOM(self.estudios_disponibles[nombre])
        if not estudio.cargar_dicom():
            print("No se pudo cargar el estudio:", nombre)
            return False

        self.estudios_cargados[nombre] = estudio
        self.estudio_actual = estudio
        self.nombre_actual = nombre
        return True