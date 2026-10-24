from clases import EstudioDICOM
estudio= EstudioDICOM("datos/T2")
estudio.cargar_dicom()
estudio.reconstruir_volumen()
estudio.corregirAngulo(30)

