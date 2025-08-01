import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score

# Ruta de los JSON
carpeta_json = "/Users/isaizurita/ProyectoTerminal/JSONS"
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + glob.glob(os.path.join(carpeta_json, "*.JSON"))
print(f"Se encontraron {len(archivos_json)} archivos JSON")

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
                        valence = float(paso["sliderDissatisfiedSatisfied"])
                        arousal = float(paso["sliderBoredExcited"])
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

    # Eliminar combinaciones duplicadas de time, risk y arrival
    df = df.drop_duplicates(subset=["time", "risk", "arrival"], keep="first")

    X = df[["time", "risk", "arrival"]]
    y = df[["valence", "arousal"]]

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Modelo KNN para regresión multisalida
    knn_regressor = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=2))
    knn_regressor.fit(X_train, y_train)

    # Predicción
    y_pred = knn_regressor.predict(X_test)

    # Redondear al valor más cercano entre 0, 25, 50, 75, 100
    def redondear_personalizado(valor):
        opciones = [0, 25, 50, 75, 100]
        return min(opciones, key=lambda x: abs(x - valor))

    # Aplicar redondeo a cada predicción
    y_pred_redondeado = []
    for pred in y_pred:
        valence_red = redondear_personalizado(pred[0])
        arousal_red = redondear_personalizado(pred[1])
        y_pred_redondeado.append([valence_red, arousal_red])
    y_pred_redondeado = pd.DataFrame(y_pred_redondeado, columns=["valence", "arousal"])

    # Métricas (usando los valores originales sin redondear para evaluación objetiva)
    #print("\n=== MÉTRICAS DE REGRESIÓN ===")
    #print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred))
    #print("Coeficiente de determinación (R^2):", r2_score(y_test, y_pred))

    # DataFrame de salida con predicciones redondeadas
    resultados = pd.DataFrame({
        "time": X_test["time"].values,
        "risk": X_test["risk"].values,
        "arrival": X_test["arrival"].values,
        "valence_predicho": y_pred_redondeado["valence"].values,
        "arousal_predicho": y_pred_redondeado["arousal"].values
    })

    print("\nResultados de la prueba:")
    print(resultados.head())

# === PRUEBA MANUAL DE UNA INSTANCIA ===

# Valores de entrada personalizados
time_input = 0.4984615284837905
risk_input = 0.6634833487600097
arrival_input = 0.19823301279341907

# Crear DataFrame con los valores de entrada
instancia_nueva = pd.DataFrame([[time_input, risk_input, arrival_input]],
                               columns=["time", "risk", "arrival"])

# Obtener predicción
prediccion = knn_regressor.predict(instancia_nueva)[0]

# Aplicar redondeo personalizado
valence_pred = redondear_personalizado(prediccion[0])
arousal_pred = redondear_personalizado(prediccion[1])

print(f"\nEntrada: time={time_input}, risk={risk_input}, arrival={arrival_input}")
print(f"Predicción -> valence: {valence_pred}, arousal: {arousal_pred}\n")
