import json  # Para leer JSONS
import glob
import os
import random
import unicodedata
import pandas as pd  # Para manejar DataFrames
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Necesitamos normalizar datos porque KNN calcula distancias,
# si las variables están en escalas distintas, algunas influirían más que otras

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

# Género no tiene un orden real entre sus categorías, así que en vez
# de darle un número ordinal arbitrario lo codificamos con one-hot:
# una columna binaria por categoría.
GENERO_CATEGORIAS = ["Male", "Female", "Other"]

def codificarGeneroOneHot(texto):
    if texto not in GENERO_CATEGORIAS:
        texto = "Other"
    return {f"genero_{cat}": 1.0 if texto == cat else 0.0 for cat in GENERO_CATEGORIAS}

# Codificamos nivel educativo en valor numérico
def codificarEducacion(texto):
    mapa = {
        "Degree": 0,
        "Master": 1,
        "Doctorate": 2
    }
    return mapa.get(texto, 0)

def quitarAcentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def nombreAIniciales(nombre):
    tokens = quitarAcentos(str(nombre)).strip().split()
    return ''.join(t[0].upper() for t in tokens if t)

# Los 10 ítems del TIPI y los 9 del ATI, en el orden del formulario
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

TIPI_SELECCION_IDX = [0, 2, 4, 6, 8]   # ítems 1, 3, 5, 7, 9
ATI_SELECCION_IDX = [1, 5, 8]          # ítems 2, 6, 9

TIPI_COLS_SELECCIONADOS = [TIPI_COLS[i] for i in TIPI_SELECCION_IDX]
ATI_COLS_SELECCIONADOS = [ATI_COLS[i] for i in ATI_SELECCION_IDX]

# Los 8 valores crudos de personalidad seleccionados
COLUMNAS_PERSONALIDAD = [f"tipi_{i + 1}" for i in TIPI_SELECCION_IDX] + \
                         [f"ati_{i + 1}" for i in ATI_SELECCION_IDX]

def construirMapaPersonalidad(rutaExcel, inicialesPorJson):
    """Devuelve {rutaJson: [8 valores]} solo para los JSON con una
    coincidencia única en el excel de personalidad."""
    dfExcel = pd.read_excel(rutaExcel, sheet_name=0)
    dfExcel = dfExcel.dropna(subset=["Ingresa tu nombre"])

    mapa = {}

    for _, fila in dfExcel.iterrows():
        nombre = fila["Ingresa tu nombre"]
        clave = nombreAIniciales(nombre)

        valores = fila[TIPI_COLS_SELECCIONADOS + ATI_COLS_SELECCIONADOS].tolist()
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
        else:
            continue

        mapa[elegido] = [float(v) for v in valores]

    return mapa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpetaJson = os.path.join(BASE_DIR, "..", "JSONS4")
rutaExcel = os.path.join(BASE_DIR, "..", "testPersonalidad.xlsx")

archivosJson = glob.glob(os.path.join(carpetaJson, "*.json")) + \
               glob.glob(os.path.join(carpetaJson, "*.JSON"))

# Primera pasada: cargamos los JSON válidos y sus iniciales para
# poder emparejarlos con el excel de personalidad
jsonCargados = {}
inicialesPorJson = {}

for archivo in archivosJson:
    with open(archivo, 'r', encoding='utf-8') as f:
        try:
            jsonData = json.load(f)
        except json.JSONDecodeError:
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

datos = []  # Lista vacía en donde guardaremos todas las muestras obtenidas

# Recorremos cada archivo para extraer información, pero solo de los
# participantes que sí respondieron el test de personalidad, los que
# no lo respondieron se omiten
for archivo, jsonData in jsonCargados.items():
    personalidad = mapaPersonalidad.get(archivo)
    if personalidad is None:
        continue

    profile = jsonData.get("profile", {})
    level = jsonData.get("level", {})

    edad = int(profile.get("age", 0))
    generoOneHot = codificarGeneroOneHot(profile.get("gender", "Other"))
    educacion = codificarEducacion(profile.get("educationLevel", "Degree"))

    familiaridad = codificarFamiliaridad(
        level.get("familiarityLevel", "")
    )

    conocimiento = codificarConocimiento(
        level.get("knowledgeLevel", "")
    )

    historias = jsonData.get("History_Experiment_Simulated", [])
    for ronda in historias:
        for paso in ronda:
            try:
                fila = {
                    "time": float(paso["time"]),
                    "risk": float(paso["risk"]),
                    "arrival": float(paso["arrival"]),
                    "age": edad,
                    "education": educacion,
                    "nivelFamiliaridad": familiaridad,
                    "nivelConocimiento": conocimiento,
                    "valence": float(paso["sliderDissatisfiedSatisfied"]),
                    "arousal": float(paso["sliderBoredExcited"])
                }
                fila.update(generoOneHot)
                fila.update(dict(zip(COLUMNAS_PERSONALIDAD, personalidad)))
                datos.append(fila)
            except Exception as e:
                print(f"Error en {archivo}: {e}")

df = pd.DataFrame(datos)

if df.empty:
    print("No se extrajeron datos")
    exit()

df = df.drop_duplicates()

X = df[[
    "time", "risk", "arrival",
    "age", "genero_Male", "genero_Female", "genero_Other", "education",
    "nivelFamiliaridad", "nivelConocimiento"
] + COLUMNAS_PERSONALIDAD]

y = df[["valence", "arousal"]]

scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

Xtrain, Xtest, yTrain, yTest = train_test_split(
    Xscaled, y, test_size=0.2, random_state=42, shuffle=False
)

knnRegressor = MultiOutputClassifier(
    KNeighborsClassifier(n_neighbors=5)
)

knnRegressor.fit(Xtrain, yTrain)

print("Accuracy of K-NN classifier on test set: {:.2f}"
      .format(knnRegressor.score(Xtest, yTest)))

# ============================================================
# Se apartan 10 puntos al azar del conjunto de prueba, simulando
# que son 10 usuarios nuevos de los que queremos predecir su reacción
# ============================================================
random.seed(42)
indicesAzar = random.sample(range(len(Xtest)), 10)

XtestMuestra = Xtest[indicesAzar]
yTestMuestra = yTest.iloc[indicesAzar].reset_index(drop=True)

prediccionMuestra = knnRegressor.predict(XtestMuestra)

valenciaReal = yTestMuestra["valence"].values
arousalReal = yTestMuestra["arousal"].values
valenciaPred = prediccionMuestra[:, 0]
arousalPred = prediccionMuestra[:, 1]

print("\nReal vs Predicción de los 10 puntos seleccionados:")
for i in range(10):
    print(f"Real: (val={valenciaReal[i]}, aro={arousalReal[i]})  ->  "
          f"Predicción: (val={valenciaPred[i]}, aro={arousalPred[i]})")

# ============================================================
# Método 1: espacio afectivo (valencia en X, arousal en Y)
# Real en azul, predicción en rojo, línea punteada uniendo cada par
# ============================================================
plt.figure(figsize=(7, 7))
plt.scatter(valenciaReal, arousalReal, color="blue", label="Real", s=80, zorder=3)
plt.scatter(valenciaPred, arousalPred, color="red", label="Predicción", s=80, zorder=3)

for i in range(10):
    plt.plot(
        [valenciaReal[i], valenciaPred[i]],
        [arousalReal[i], arousalPred[i]],
        linestyle="--", color="gray", zorder=1
    )

plt.xlabel("Valencia")
plt.ylabel("Arousal")
plt.title("Espacio afectivo: real (azul) vs predicción (rojo)")
plt.xlim(-5, 105)
plt.ylim(-5, 105)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "grafica_espacio_afectivo.png"), dpi=150)

# ============================================================
# Método 2: línea de identidad, por separado para valencia y arousal
# Eje X = valor real, eje Y = valor predicho, diagonal a 45 grados
# ============================================================
fig, ejes = plt.subplots(1, 2, figsize=(12, 6))

datosPorEmocion = [
    ("Valencia", valenciaReal, valenciaPred),
    ("Arousal", arousalReal, arousalPred),
]

for ax, (nombre, real, pred) in zip(ejes, datosPorEmocion):
    ax.scatter(real, pred, color="blue", s=80, zorder=3)
    ax.plot([0, 100], [0, 100], linestyle="--", color="black", zorder=1)
    ax.set_xlabel(f"{nombre} real")
    ax.set_ylabel(f"{nombre} predicha")
    ax.set_title(f"Línea de identidad - {nombre}")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "grafica_linea_identidad.png"), dpi=150)

plt.show()
