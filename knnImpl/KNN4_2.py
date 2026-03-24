import json
import glob
import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np

def codificar_familiaridad(texto):
    mapa = {
        "Never seen a virtual or physical service robot": 0,
        "Seen physical service robots": 1,
        "Seen virtual service robots": 2,
        "Seen both physical and virtual service robots": 3
    }
    return mapa.get(texto, 0)

def codificar_conocimiento(texto):
    mapa = {
        "Never programmed robots": 0,
        "Programmed physical robots": 1,
        "Programmed virtual robots": 1,
        "Programmed both physical and virtual robots": 2
    }
    return mapa.get(texto, 0)

def codificar_genero(texto):
    mapa = {
        "Male": 0,
        "Female": 1,
        "Other": 2
    }
    return mapa.get(texto, 2)

def codificar_educacion(texto):
    mapa = {
        "Degree": 0,
        "Master": 1,
        "Doctorate": 2
    }
    return mapa.get(texto, 0)

def redondear_personalizado(valor):
    opciones = [0, 25, 50, 75, 100]
    return min(opciones, key=lambda x: abs(x - valor))

carpeta_json = "/Users/isaizurita/emocionesJSON/JSONS4"
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + \
                glob.glob(os.path.join(carpeta_json, "*.JSON"))

print(f"Se encontraron {len(archivos_json)} archivos JSON")

datos = []

for archivo in archivos_json:
    with open(archivo, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)

            profile = json_data.get("profile", {})
            level = json_data.get("level", {})

            edad = int(profile.get("age", 0))
            genero = codificar_genero(profile.get("gender", "Other"))
            educacion = codificar_educacion(profile.get("educationLevel", "Degree"))

            familiaridad = codificar_familiaridad(
                level.get("familiarityLevel", "")
            )

            conocimiento = codificar_conocimiento(
                level.get("knowledgeLevel", "")
            )

            historias = json_data.get("History_Experiment_Simulated", [])

            for ronda in historias:
                for paso in ronda:
                    try:
                        datos.append({
                            "time": float(paso["time"]),
                            "risk": float(paso["risk"]),
                            "arrival": float(paso["arrival"]),
                            "age": edad,
                            "gender": genero,
                            "education": educacion,
                            "nivelFamiliaridad": familiaridad,
                            "nivelConocimiento": conocimiento,
                            "valence": float(paso["sliderDissatisfiedSatisfied"]),
                            "arousal": float(paso["sliderBoredExcited"])
                        })
                    except Exception as e:
                        print(f"Error en {archivo}: {e}")

        except json.JSONDecodeError as e:
            print(f"Error leyendo {archivo}: {e}")

df = pd.DataFrame(datos)

if df.empty:
    print("No se extrajeron datos")
    exit()

print(f"Se extrajeron {len(df)} muestras")

df = df.drop_duplicates()

X = df[[
    "time", "risk", "arrival",
    "age", "gender", "education",
    "nivelFamiliaridad", "nivelConocimiento"
]]

y = df[["valence", "arousal"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

knn_regressor = MultiOutputRegressor(
    KNeighborsRegressor(n_neighbors=3)
)

knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

y_pred_redondeado = [
    [
        redondear_personalizado(pred[0]),
        redondear_personalizado(pred[1])
    ]
    for pred in y_pred
]

print("\nPrimeras predicciones:")
print(pd.DataFrame(y_pred_redondeado, columns=["valence", "arousal"]).head())


nueva_instancia = pd.DataFrame([{
    "time": 0.49,
    "risk": 0.66,
    "arrival": 0.19,
    "age": 44,
    "gender": codificar_genero("Female"),
    "education": codificar_educacion("Master"),
    "nivelFamiliaridad": codificar_familiaridad(
        "Seen both physical and virtual service robots"
    ),
    "nivelConocimiento": codificar_conocimiento(
        "Never programmed robots"
    )
}])

nueva_instancia_scaled = scaler.transform(nueva_instancia)

prediccion = knn_regressor.predict(nueva_instancia_scaled)[0]

valence_pred = redondear_personalizado(prediccion[0])
arousal_pred = redondear_personalizado(prediccion[1])

print("\nPredicción nueva instancia:")
print("Valence:", valence_pred)
print("Arousal:", arousal_pred)

X_simple = df[["time", "risk", "arrival"]]
y = df[["valence", "arousal"]]

scaler_simple = StandardScaler()
X_simple_scaled = scaler_simple.fit_transform(X_simple)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple_scaled, y, test_size=0.2, random_state=42
)

modelo_simple = MultiOutputRegressor(
    KNeighborsRegressor(n_neighbors=17)
)

modelo_simple.fit(X_train_s, y_train_s)

y_pred_s = modelo_simple.predict(X_test_s)

y_pred_s_redondeado = [
    [
        redondear_personalizado(pred[0]),
        redondear_personalizado(pred[1])
    ]
    for pred in y_pred_s
]

y_pred_s_redondeado = pd.DataFrame(
    y_pred_s_redondeado,
    columns=["valence", "arousal"]
)

mae_valence_simple = mean_absolute_error(
    y_test_s["valence"], y_pred_s_redondeado["valence"]
)

mae_arousal_simple = mean_absolute_error(
    y_test_s["arousal"], y_pred_s_redondeado["arousal"]
)

print("\nModelo SIMPLE (solo variables técnicas)")
print("MAE Valence:", mae_valence_simple)
print("MAE Arousal:", mae_arousal_simple)

print("\nComparación Real vs Predicho (Simple):")
comparacion_simple = pd.DataFrame({
    "Valence_Real": y_test_s["valence"].values,
    "Valence_Pred": y_pred_s_redondeado["valence"].values,
    "Arousal_Real": y_test_s["arousal"].values,
    "Arousal_Pred": y_pred_s_redondeado["arousal"].values
})
print(comparacion_simple.head())


# -------------------------------
# MODELO 2: variables completas
# -------------------------------

X_full = df[[
    "time", "risk", "arrival",
    "age", "gender", "education",
    "nivelFamiliaridad", "nivelConocimiento"
]]

scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full_scaled, y, test_size=0.2, random_state=42
)

modelo_full = MultiOutputRegressor(
    KNeighborsRegressor(n_neighbors=5)
)

modelo_full.fit(X_train_f, y_train_f)

y_pred_f = modelo_full.predict(X_test_f)

y_pred_f_redondeado = [
    [
        redondear_personalizado(pred[0]),
        redondear_personalizado(pred[1])
    ]
    for pred in y_pred_f
]

y_pred_f_redondeado = pd.DataFrame(
    y_pred_f_redondeado,
    columns=["valence", "arousal"]
)

mae_valence_full = mean_absolute_error(
    y_test_f["valence"], y_pred_f_redondeado["valence"]
)

mae_arousal_full = mean_absolute_error(
    y_test_f["arousal"], y_pred_f_redondeado["arousal"]
)

print("\nModelo COMPLETO (con variables demográficas)")
print("MAE Valence:", mae_valence_full)
print("MAE Arousal:", mae_arousal_full)

print("\nComparación Real vs Predicho (Completo):")
comparacion_full = pd.DataFrame({
    "Valence_Real": y_test_f["valence"].values,
    "Valence_Pred": y_pred_f_redondeado["valence"].values,
    "Arousal_Real": y_test_f["arousal"].values,
    "Arousal_Pred": y_pred_f_redondeado["arousal"].values
})
print(comparacion_full.head())

X = df[["time", "risk", "arrival"]]
y = df[["valence", "arousal"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mae_scorer = make_scorer(mean_absolute_error)

k_values = range(1, 31)
resultados = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    modelo = MultiOutputRegressor(
        KNeighborsRegressor(n_neighbors=k)
    )
    
    scores = cross_val_score(
        modelo,
        X_scaled,
        y,
        cv=kf,
        scoring=mae_scorer
    )
    
    resultados.append({
        "k": k,
        "MAE_promedio": np.mean(scores)
    })

resultados_df = pd.DataFrame(resultados)

print("\nResultados validación cruzada:")
print(resultados_df)

mejor_fila = resultados_df.loc[resultados_df["MAE_promedio"].idxmin()]

print("\nMejor k encontrado:")
print("k =", int(mejor_fila["k"]))
print("MAE promedio =", mejor_fila["MAE_promedio"])