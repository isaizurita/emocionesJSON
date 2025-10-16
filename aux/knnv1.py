import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Ruta donde están los archivos JSON
carpeta_json = "/Users/isaizurita/Downloads/JSONS"

# Lista para almacenar todos los datos
datos = []

# Leer todos los archivos JSON en la carpeta
for archivo in os.listdir(carpeta_json):
    if archivo.endswith(".json"):
        ruta = os.path.join(carpeta_json, archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            try:
                contenido = json.load(f)
                
                # Si el archivo es una lista de rondas directamente
                if isinstance(contenido, list):
                    rondas = contenido
                # Si es un diccionario con clave "rondas"
                elif isinstance(contenido, dict) and "rondas" in contenido:
                    rondas = contenido["rondas"]
                else:
                    print(f"Formato no esperado en archivo: {archivo}")
                    continue

                for i, ronda in enumerate(rondas):
                    datos.append({
                        "usuario": archivo,
                        "ronda": i + 1,
                        "valence": ronda.get("valence"),
                        "arousal": ronda.get("arousal")
                    })

            except json.JSONDecodeError:
                print(f"Error al leer archivo: {archivo}")

# Crear DataFrame con los datos
df = pd.DataFrame(datos)

# Mostrar cuántas rondas tiene cada usuario
print(df.groupby("usuario")["ronda"].count())

# Mostrar estadística descriptiva
print(df.describe())

# Gráfica de dispersión
plt.figure(figsize=(10, 6))
for usuario in df["usuario"].unique():
    subset = df[df["usuario"] == usuario]
    plt.scatter(subset["valence"], subset["arousal"], label=usuario)

plt.xlabel("Valencia")
plt.ylabel("Excitación")
plt.title("Emociones por ronda y usuario")
plt.legend()
plt.grid(True)
plt.show()
