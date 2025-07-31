import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ruta de los JSON
carpeta_json = "/Users/isaizurita/ProyectoTerminal/JSONS"
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + glob.glob(os.path.join(carpeta_json, "*.JSON"))
print(f"\nSe encontraron {len(archivos_json)} archivos JSON")

datos = []

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
                        datos.append({
                            "time": time,
                            "risk": risk,
                            "arrival": arrival,
                            "valence": valence,
                            "arousal": arousal
                        })
                    except Exception as e:
                        print(f"Error al leer un paso en {archivo}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error leyendo {archivo}: {e}")

df = pd.DataFrame(datos)

if df.empty:
    print("No se extrajeron datos")
else:
    print(f"\nSe extrajeron {len(df)} muestras de entrenamiento")

    # Variables predictoras (X) y etiquetas (y)
    X = df[["time", "risk", "arrival"]]
    y = df[["valence", "arousal"]]

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo KNN para clasificación multisalida
    knn_classifier = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
    knn_classifier.fit(X_train, y_train)

    # Predicción
    y_pred = knn_classifier.predict(X_test)

    # Métricas por variable
    print("\n=== MÉTRICAS DE CLASIFICACIÓN ===")

    print("\nVALENCE:")
    print(classification_report(y_test["valence"], y_pred[:, 0]))

    print("\nAROUSAL:")
    print(classification_report(y_test["arousal"], y_pred[:, 1]))

    # DataFrame de salida con time, valence_predicho y arousal_predicho
    resultados = pd.DataFrame({
        "time": X_test["time"].values,
        "valence_predicho": y_pred[:, 0],
        "arousal_predicho": y_pred[:, 1]
    })

    print("\nPrimeras filas del DataFrame de resultados:")
    print(resultados.head())
