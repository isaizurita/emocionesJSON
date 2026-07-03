import json # Para leer JSONS
import glob
import os
import unicodedata
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

def quitarAcentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def nombreAIniciales(nombre):
    tokens = quitarAcentos(str(nombre)).strip().split()
    return ''.join(t[0].upper() for t in tokens if t)

TIPI_COLS = [
    "Extrovertida/Entusiasta",
    "Colérica/Discutidora",
    "Fiable/Auto-Disciplinada",
    "Ansiosa/Fácilmente alterable",
    "Abierta a nuevas experiencias/Polifacética",
    "Reservada/Callada",
    "Comprensiva/Amable",
    "Desorganizada/descuidada",
    "Serena/Emocionalmente estable",
    "Tradicional/Poco imaginativa",
]

ATI_COLS = [
    "Me gusta ocuparme en mayor detalle de los sistemas técnicos",
    "Me gusta probar las funciones de los nuevos sistemas técnicos",
    "Principalmente trato con sistemas técnicos porque tengo que hacerlo.",
    "Cuando tengo un nuevo sistema técnico frente a mí, lo pruebo intensamente",
    "Me gusta pasar tiempo familiarizándome con un nuevo sistema técnico",
    "Para mí es suficiente que funcione un sistema técnico, no me importa ni el cómo ni el por qué",
    "Intento entender cómo funciona exactamente un sistema técnico",
    "Para mí es suficiente conocer las funciones básicas de un sistema técnico",
    "Intento aprovechar al máximo las capacidades de un sistema técnico",
]

# Los 19 valores crudos del test de personalidad
COLUMNAS_PERSONALIDAD = [f"tipi_{i + 1}" for i in range(10)] + \
                         [f"ati_{i + 1}" for i in range(9)]

def construirMapaPersonalidad(rutaExcel, inicialesPorJson):
    """Devuelve {rutaJson: [19 valores]} solo para los JSON con una
    coincidencia única en el excel de personalidad."""
    dfExcel = pd.read_excel(rutaExcel, sheet_name=0)
    dfExcel = dfExcel.dropna(subset=["Ingresa tu nombre"])

    mapa = {}
    sinMatch = []
    ambiguos = []

    for _, fila in dfExcel.iterrows():
        nombre = fila["Ingresa tu nombre"]
        clave = nombreAIniciales(nombre)

        valores = fila[TIPI_COLS + ATI_COLS].tolist()
        if any(pd.isna(v) for v in valores):
            continue

        candidatos = [
            ruta for ruta, ini in inicialesPorJson.items()
            if ini.startswith(clave)
        ]
        exactos = [r for r in candidatos if inicialesPorJson[r] == clave]

        if exactos:
            elegido = exactos[0]
        elif len(candidatos) == 1:
            elegido = candidatos[0]
        elif len(candidatos) == 0:
            sinMatch.append(nombre)
            continue
        else:
            ambiguos.append((nombre, candidatos))
            continue

        mapa[elegido] = [float(v) for v in valores]

    return mapa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpetaJson = os.path.join(BASE_DIR, "..", "JSONS4")
rutaExcel = os.path.join(BASE_DIR, "..", "testPersonalidad.xlsx")

archivosJson = glob.glob(os.path.join(carpetaJson, "*.json")) + \
               glob.glob(os.path.join(carpetaJson, "*.JSON"))

print(f"\nSe encontraron {len(archivosJson)} archivos JSON")

# Primera pasada: cargamos los JSON válidos y sus iniciales para
# poder emparejarlos con el excel de personalidad
jsonCargados = {}
inicialesPorJson = {}

for archivo in archivosJson:
    with open(archivo, 'r', encoding='utf-8') as f:
        try:
            jsonData = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error leyendo {archivo}: {e}")
            continue

    if not isinstance(jsonData, dict) or "profile" not in jsonData:
        continue

    iniciales = quitarAcentos(
        str(jsonData.get("profile", {}).get("initials", ""))
    ).strip().upper()

    if not iniciales:
        continue

    jsonCargados[archivo] = jsonData
    inicialesPorJson[archivo] = iniciales

mapaPersonalidad = construirMapaPersonalidad(rutaExcel, inicialesPorJson)

datos = [] # Lista vacía en donde guardaremos todas las muestras obtenidas

# Recorremos cada archivo para extraer información, pero solo de los
# participantes que sí respondieron el test de personalidad los que
# no lo respondieron se omiten
for archivo, jsonData in jsonCargados.items():
    personalidad = mapaPersonalidad.get(archivo)
    if personalidad is None:
        continue

    profile = jsonData.get("profile", {}) # Extraemos las cabeceras de las nuevas variables
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
                fila = {
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
                }
                fila.update(dict(zip(COLUMNAS_PERSONALIDAD, personalidad))) # Valores fijos de personalidad
                datos.append(fila)
            except Exception as e:
                print(f"Error en {archivo}: {e}")

# Convertimos la lista en DataFrame
df = pd.DataFrame(datos)

if df.empty:
    print("No se extrajeron datos")
    exit()

# Eliminamos duplicados exactos
df = df.drop_duplicates()

# Definimos nuestras "x y y"
# Variables de entrada: las técnicas + demográficas de siempre, más los 19 valores de personalidad
X = df[[
    "time", "risk", "arrival",
    "age", "gender", "education",
    "nivelFamiliaridad", "nivelConocimiento"
] + COLUMNAS_PERSONALIDAD]

# Variables objetivo
y = df[["valence", "arousal"]]

# Estandarizamos todas las variables para que todas tengan la misma escala, esto las pone en igualdad
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

# División en entrenamiento y prueba, 80% entrenamiento y 20% prueba
Xtrain, Xtest, yTrain, yTest = train_test_split(
    Xscaled, y, test_size=0.2, random_state=42, shuffle=False
)

# Creación del modelo multisalida (valence, arousal)
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

INICIALES_EJEMPLO = "ARNR"

archivoEjemplo = next(
    a for a, ini in inicialesPorJson.items() if ini == INICIALES_EJEMPLO
)
jsonEjemplo = jsonCargados[archivoEjemplo]
profileEjemplo = jsonEjemplo.get("profile", {})
levelEjemplo = jsonEjemplo.get("level", {})
personalidadEjemplo = mapaPersonalidad[archivoEjemplo]  # <- los 19 valores reales de TIPI + ATI

pasoEjemplo = jsonEjemplo["History_Experiment_Simulated"][0][0]

nuevaInstancia = pd.DataFrame([{
    "time": float(pasoEjemplo["time"]),
    "risk": float(pasoEjemplo["risk"]),
    "arrival": float(pasoEjemplo["arrival"]),
    "age": int(profileEjemplo.get("age", 0)),
    "gender": codificarGenero(profileEjemplo.get("gender", "Other")),
    "education": codificarEducacion(profileEjemplo.get("educationLevel", "Degree")),
    "nivelFamiliaridad": codificarFamiliaridad(
        levelEjemplo.get("familiarityLevel", "")
    ),
    "nivelConocimiento": codificarConocimiento(
        levelEjemplo.get("knowledgeLevel", "")
    ),
    **dict(zip(COLUMNAS_PERSONALIDAD, personalidadEjemplo))
}])

# Aplicamos el mismo escalado que el entrenamiento
nuevaInstanciaScaled = scaler.transform(nuevaInstancia)

prediccion = knnRegressor.predict(nuevaInstanciaScaled)[0]

valencePred = redondearPersonalizado(prediccion[0])
arousalPred = redondearPersonalizado(prediccion[1])

print(f"\nInstancia real: {INICIALES_EJEMPLO}")

for columna in nuevaInstancia.columns:
    print(f"{columna:20}: {nuevaInstancia[columna][0]}")

print("\nPredicción del modelo:")
print(f"Valence : {valencePred}")
print(f"Arousal : {arousalPred}")

print("\nRespuesta real de la persona en ese paso:")
print(f"Valence : {pasoEjemplo['sliderDissatisfiedSatisfied']}")
print(f"Arousal : {pasoEjemplo['sliderBoredExcited']}")

#======================================
pred = knnRegressor.predict(Xtest)

print("\nConfusion Matrix - Valence")
print(confusion_matrix(yTest["valence"], pred[:,0]))

print("\nClassification Report - Valence")
print(classification_report(yTest["valence"], pred[:,0], zero_division=0))

print("\nConfusion Matrix - Arousal")
print(confusion_matrix(yTest["arousal"], pred[:,1]))

print("\nClassification Report - Arousal")
print(classification_report(yTest["arousal"], pred[:,1], zero_division=0))
