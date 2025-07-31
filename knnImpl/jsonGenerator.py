import json
import random
import math

def generar_json(num_pares, archivo_salida):
    data = {
        "History_Experiment_Simulated": []
    }
    
    # Valores permitidos para valence y arousal
    opcionesSlider = [0, 25, 50, 75, 100]

    for i in range(num_pares):
        par = []

        # Generar parámetros de entrada para cada paso
        time_0 = random.uniform(0.4, 0.7)
        risk_0 = random.uniform(0.6, 1.0)
        arrival_0 = random.uniform(0.0, 0.3)
        
        time_1 = random.uniform(0.4, 0.7)
        risk_1 = random.uniform(0.6, 1.0)
        arrival_1 = random.uniform(0.0, 0.3)
        
        # Asignar valores aleatorios de la lista a valence y arousal
        valence_0 = random.choice(opcionesSlider)
        arousal_0 = random.choice(opcionesSlider)
        
        valence_1 = random.choice(opcionesSlider)
        arousal_1 = random.choice(opcionesSlider)

        par.append({
            "id": 0,
            "time": round(time_0, 6),
            "risk": round(risk_0, 6),
            "arrival": round(arrival_0, 6),
            "sliderDissatisfiedSatisfied": str(valence_0),
            "sliderBoredExcited": str(arousal_0)
        })

        par.append({
            "id": 1,
            "time": round(time_1, 6),
            "risk": round(risk_1, 6),
            "arrival": round(arrival_1, 6),
            "sliderDissatisfiedSatisfied": str(valence_1),
            "sliderBoredExcited": str(arousal_1)
        })

        data["History_Experiment_Simulated"].append(par)

    with open(archivo_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Archivo {archivo_salida} generado con {num_pares * 2} muestras (en {num_pares} pares).")

# Generar archivo con los nuevos valores
generar_json(10000, "datos_prueba_relacionados.json")