# Versión con redondeo
import json
import glob
import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# Ruta de los JSON
carpeta_json = "/Users/isaizurita/emocionesJSON/JSONS2"
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + \
                glob.glob(os.path.join(carpeta_json, "*.JSON"))
print(f"Se encontraron {len(archivos_json)} archivos JSON")

datos = []

for archivo in archivos_json:
    with open(archivo, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)

            # Normalizamos a una lista de historias
            historias = []

            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict) and "History_Experiment_Simulated" in item:
                        historias.append(item["History_Experiment_Simulated"])
            elif isinstance(json_data, dict):
                historias.append(json_data.get("History_Experiment_Simulated", []))

            # Procesamiento normal
            for historia in historias:
                for ronda in historia:
                    for paso in ronda:
                        try:
                            datos.append({
                                "time": float(paso["time"]),
                                "risk": float(paso["risk"]),
                                "arrival": float(paso["arrival"]),
                                "valence": float(paso["sliderDissatisfiedSatisfied"]),
                                "arousal": float(paso["sliderBoredExcited"])
                            })
                        except Exception as e:
                            print(f"Error al leer un paso en {archivo}: {e}")

        except json.JSONDecodeError as e:
            print(f"Error leyendo {archivo}: {e}")

# DataFrame
df = pd.DataFrame(datos)

if df.empty:
    print("No se extrajeron datos")
    exit()

print(f"\nSe extrajeron {len(df)} muestras de entrenamiento")

# Eliminar combinaciones duplicadas
df = df.drop_duplicates(subset=["time", "risk", "arrival"], keep="first")

X = df[["time", "risk", "arrival"]]
y = df[["valence", "arousal"]]

# División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Modelo KNN 
knn_regressor = MultiOutputRegressor(
    KNeighborsRegressor(n_neighbors=2)
)
knn_regressor.fit(X_train, y_train)

# Predicción
y_pred = knn_regressor.predict(X_test)

# Redondeo personalizado
def redondear_personalizado(valor):
    opciones = [0, 25, 50, 75, 100]
    return min(opciones, key=lambda x: abs(x - valor))

y_pred_redondeado = [
    [
        redondear_personalizado(pred[0]),
        redondear_personalizado(pred[1])
    ]
    for pred in y_pred
]

y_pred_redondeado = pd.DataFrame(
    y_pred_redondeado,
    columns=["valence", "arousal"]
)

# Resultados
resultados = pd.DataFrame({
    "time": X_test["time"].values,
    "risk": X_test["risk"].values,
    "arrival": X_test["arrival"].values,
    "valence_predicho": y_pred_redondeado["valence"].values,
    "arousal_predicho": y_pred_redondeado["arousal"].values
})

print("\nResultados de la prueba:")
print(resultados.head())

# Prueba manual
time_input = 0.4984615284837905
risk_input = 0.6634833487600097
arrival_input = 0.19823301279341907

instancia_nueva = pd.DataFrame(
    [[time_input, risk_input, arrival_input]],
    columns=["time", "risk", "arrival"]
)

prediccion = knn_regressor.predict(instancia_nueva)[0]

valence_pred = redondear_personalizado(prediccion[0])
arousal_pred = redondear_personalizado(prediccion[1])

print(f"\nEntrada: time={time_input}, risk={risk_input}, arrival={arrival_input}")
print(f"Predicción -> valence: {valence_pred}, arousal: {arousal_pred}\n")
