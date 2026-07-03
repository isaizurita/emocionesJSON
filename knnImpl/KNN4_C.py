import json # Para leer JSONS
import glob # """
import os
import pandas as pd # Para manejar DataFrames
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Necesitamos normalizar datos porque KNN calcula distancias, 
# si las variables están en escalas distintas, algunas influirían más que otras
from sklearn.metrics import confusion_matrix, classification_report

# Codificamos el nivel de familiaridad con robots en un valor numérico usando diccionarios
def codificarFamiliaridad(texto):
    mapa = {
        "Never seen a virtual or physical service robot": 0,
        "Seen physical service robots": 1,
        "Seen virtual service robots": 2,
        "Seen both physical and virtual service robots": 3
    }
    return mapa.get(texto, 0)

# Codificamos el nivel de conocimiento/programación en robots
def codificarConocimiento(texto):
    mapa = {
        "Never programmed robots": 0,
        "Programmed physical robots": 1,
        "Programmed virtual robots": 1,
        "Programmed both physical and virtual robots": 2
    }
    return mapa.get(texto, 0)

# Codificamos género en valor numérico
def codificarGenero(texto):
    mapa = {
        "Male": 0,
        "Female": 1,
        "Other": 2
    }
    return mapa.get(texto, 2)

# Codificamos nivel educativo en valor numérico
def codificarEducacion(texto):
    mapa = {
        "Degree": 0,
        "Master": 1,
        "Doctorate": 2
    }
    return mapa.get(texto, 0)

# Se redondea la emoción al nivel permitido más cercano
def redondearPersonalizado(valor):
    opciones = [0, 25, 50, 75, 100]
    return min(opciones, key=lambda x: abs(x - valor))

carpetaJson = "/Users/isaizurita/emocionesJSON/JSONS4"

archivosJson = glob.glob(os.path.join(carpetaJson, "*.json")) + \
               glob.glob(os.path.join(carpetaJson, "*.JSON"))

print(f"\nSe encontraron {len(archivosJson)} archivos JSON")

datos = [] #Lista vacía en donde guardaaremos todas las muestras obtenidas

# Recorremos cada archivo para extraer información
for archivo in archivosJson:
    with open(archivo, 'r', encoding='utf-8') as f:
        try:
            jsonData = json.load(f) # Convertimos el JSON a diccionario

            if not isinstance(jsonData, dict) or "profile" not in jsonData:
                continue # Archivos auxiliares (p. ej. rondas_usuarios.json) sin estructura de participante

            profile = jsonData.get("profile", {}) #Extraemos las cabeceras de las nuevas variables
            level = jsonData.get("level", {})

            # Extraemos tal cual las nuevas variables
            edad = int(profile.get("age", 0))
            genero = codificarGenero(profile.get("gender", "Other"))
            educacion = codificarEducacion(profile.get("educationLevel", "Degree"))

            # Mandamos a llamar a las funciones para estandarizar la familiaridad y el nivel de conocimiento
            familiaridad = codificarFamiliaridad(
                level.get("familiarityLevel", "")
            )

            conocimiento = codificarConocimiento(
                level.get("knowledgeLevel", "")
            )

            # Recorremos todas las rondas y pasos del experimento
            historias = jsonData.get("History_Experiment_Simulated", [])
            for ronda in historias:
                for paso in ronda:
                    try:
                        datos.append({
                            "time": float(paso["time"]),
                            "risk": float(paso["risk"]),
                            "arrival": float(paso["arrival"]),
                            "age": edad, # Valores fijos para todos los pasos de esta persona
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

# Convertimos la lista en DataFrame
df = pd.DataFrame(datos)

if df.empty:
    print("No se extrajeron datos")
    exit()

print(f"Se extrajeron {len(df)} muestras")

# Eliminamos duplicados exactos
df = df.drop_duplicates()

# Definimos nuestras "x y y"
# Variables de entrada
X = df[[
    "time", "risk", "arrival",
    "age", "gender", "education",
    "nivelFamiliaridad", "nivelConocimiento"
]]

# Variables objetivo
y = df[["valence", "arousal"]]

# Estandarizamos todas las variables para que todas tengan la misma escala, esto las pone en igualdad
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

# División en entrenamiento y prueba, 80% entrenamiento y 20% prueba
Xtrain, Xtest, yTrain, yTest = train_test_split(
    Xscaled, y, test_size=0.2, random_state=42, shuffle=False
)

# Creación del modelo multidalida (valence, arousal)
knnRegressor = MultiOutputClassifier(
    KNeighborsClassifier(n_neighbors=5)
)

# Entrenamiento
knnRegressor.fit(Xtrain, yTrain)

#Accuracy
#===================================
print("\nAccuracy of K-NN classifier on training set: {:.2f}"
      .format(knnRegressor.score(Xtrain, yTrain)))

print("Accuracy of K-NN classifier on test set: {:.2f}"
      .format(knnRegressor.score(Xtest, yTest)))
#===================================

# Predicción sobre el conjunto de prueba, para cada punto de prueba, busca los k=5 vecinos más cercanos
yPred = knnRegressor.predict(Xtest)

# Redondeamos emociones al formato que se acordó [0,25,50,75,100]
yPredRedondeado = [
    [
        redondearPersonalizado(pred[0]),
        redondearPersonalizado(pred[1])
    ]
    for pred in yPred
]

# Creamos una nueva instancia manual
nuevaInstancia = pd.DataFrame([{
    "time": 0.4846596000512216,
    "risk": 0.9272238212895264,
    "arrival": 0.1449310118720613,
    "age": 50,
    "gender": codificarGenero("Male"),
    "education": codificarEducacion("Doctorate"),
    "nivelFamiliaridad": codificarFamiliaridad(
        "Seen physical service robots"
    ),
    "nivelConocimiento": codificarConocimiento(
        "Never programmed robots"
    )
}])

# Aplicamos el mismo escalado que el entrenamiento
nuevaInstanciaScaled = scaler.transform(nuevaInstancia)

prediccion = knnRegressor.predict(nuevaInstanciaScaled)[0]

valencePred = redondearPersonalizado(prediccion[0])
arousalPred = redondearPersonalizado(prediccion[1])

print("\n***Instancia ingresada manualmente:")

for columna in nuevaInstancia.columns:
    print(f"{columna:20}: {nuevaInstancia[columna][0]}")

print("\nPredicción del modelo:")
print(f"Valence : {valencePred}")
print(f"Arousal : {arousalPred}")

#======================================
pred = knnRegressor.predict(Xtest)

print("\nConfusion Matrix - Valence")
print(confusion_matrix(yTest["valence"], pred[:,0]))

print("\nClassification Report - Valence")
classification_report(yTest["valence"], pred[:,0], zero_division=0)

print("\nConfusion Matrix - Arousal")
print(confusion_matrix(yTest["arousal"], pred[:,1]))

print("\nClassification Report - Arousal")
print(classification_report(yTest["arousal"], pred[:,1]))
