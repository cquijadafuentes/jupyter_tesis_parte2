import sys
import zlib
import numpy as np

def leer_archivo(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        lineas = f.readlines()
    n, m = map(int, lineas[0].split())
    return np.array([list(map(float, linea.strip().split())) for linea in lineas[2:2+n]])

def codificar_alt1_sin_cuantizacion(datos):
    """Alternativa 1 pura sin cuantización"""
    encoded = np.zeros_like(datos)
    encoded[:, 0] = datos[:, 0]
    for i in range(1, datos.shape[1]):
        encoded[:, i] = datos[:, i] - datos[:, i-1]
    return encoded

def codificar_hibrido_sin_cuantizacion(datos):
    """Híbrido sin cuantización, usando floats nativos"""
    corr_matrix = np.corrcoef(datos)
    ref_index = np.argmax(np.mean(corr_matrix, axis=1))
    referencia = datos[ref_index]
    
    # Diferencias espaciales y temporales
    dif_espacial = datos - referencia
    deltas = np.diff(dif_espacial, axis=1, prepend=0)
    return deltas

def comprimir_justo(datos_codificados):
    """Compresión usando representación binaria nativa de floats (64 bits)"""
    return zlib.compress(datos_codificados.tobytes())

def main(archivo):
    datos = leer_archivo(archivo)
    
    # Codificar
    alt1 = codificar_alt1_sin_cuantizacion(datos)
    hibrido = codificar_hibrido_sin_cuantizacion(datos)
    
    # Comprimir
    c_alt1 = len(comprimir_justo(alt1))
    c_hibrido = len(comprimir_justo(hibrido))
    
    # Resultados
    print("\n=== COMPARACIÓN JUSTA (sin cuantización) ===")
    print(f"Alternativa 1: {c_alt1} bytes")
    print(f"Híbrido: {c_hibrido} bytes")
    print(f"Diferencia: {c_alt1 - c_hibrido} bytes ({((c_alt1 - c_hibrido)/c_alt1)*100:.1f}% mejor)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python comparacion_justa.py <archivo.txt>")
        sys.exit(1)
    main(sys.argv[1])
    