import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
#KNN libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #Esta bibioteca es crucial para dividir un conjunto de datos en dos subconjuntos: uno para entrenamiento del modelo y otro para prueba
from sklearn.metrics import classification_report #Esta función genera un informe detallado de las métricas de clasificación (precisión, recall, f1-score, soporte) para evaluar el rendimiento de un modelo

#Ruta de los JSON
carpeta_json = "/Users/isaizurita/ProyectoTerminal/JSONS"

#Búsqueda de los archivos JSON
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + glob.glob(os.path.join(carpeta_json, "*.JSON"))
print(f"\nSe encontraron {len(archivos_json)} archivos JSON")

#Lista para guardar los datos
datos = []

#Recorremos cada archivo JSON
for archivo in archivos_json:
    with open(archivo, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
            historia = json_data.get("History_Experiment_Simulated", [])
            for ronda in historia:
                for paso in ronda:
                    try:
                        time = float(paso["time"])
                        risk = float(paso["risk"])
                        arrival = float(paso["arrival"])
                        valence = int(paso["sliderDissatisfiedSatisfied"])
                        arousal = int(paso["sliderBoredExcited"])
                        datos.append(
                            {
                                "time": time,
                                "risk": risk,
                                "arrival": arrival,
                                "valence": valence,
                                "arousal": arousal
                            }
                        )
                    except Exception as e:
                        print(f"Error al leer un paso en {archivo}: {e}")

        except json.JSONDecodeError as e:
            print(f"Error leyendo {archivo}: {e}")
                    
#Convertimos a DataFrame
df = pd.DataFrame(datos)

if df.empty:
    print(f"No se extrajeron datos")
else:
    print(f"\nSe extrajeron {len(df)} muestras de entrenamiento")

    #Entrenamos un modelo KNN para la valencia
    print("\n=== Predicción de VALENCIA ===")
    X = df[["risk", "arrival"]]
    y_valence = df["valence"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_valence, test_size=0.2, random_state=42)

    knn_valence = KNeighborsClassifier(n_neighbors=5)
    knn_valence.fit(X_train, y_train)
    y_pred_valence = knn_valence.predict(X_test)
    print(classification_report(y_test, y_pred_valence))

    #Entrenamos otro modelo KNN para el arousal
    print("\n=== Predicción de AROUSAL ===")
    y_arousal = df["arousal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_arousal, test_size=0.2, random_state=42)

    knn_arousal = KNeighborsClassifier(n_neighbors=5)
    knn_arousal.fit(X_train, y_train)
    y_pred_arousal = knn_arousal.predict(X_test)
    print(classification_report(y_test, y_pred_arousal))
