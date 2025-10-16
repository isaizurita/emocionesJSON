# Predicción de Emociones mediante Aprendizaje Automático
**Proyecto Terminal I – Ingeniería en Computación**  
**Autor:** Isaí Obed Zurita Prado  
**Asesores:** Dra. Alicia Montserrat Alvarado González y Dr. Antonio López Jaimes 

---

## Descripción General

Este proyecto explora el uso de **aprendizaje automático** para **predecir emociones humanas** en un entorno simulado de toma de decisiones.  
El sistema se basa en los registros emocionales de los usuarios —medidos en las dimensiones de **valencia** (agrado/desagrado) y **arousal** (nivel de activación)— para anticipar sus respuestas frente a nuevos estímulos.

El objetivo principal es **disminuir la carga cognitiva** del usuario y hacer más natural la interacción humano-máquina, especialmente en contextos donde la repetición de decisiones puede generar **fatiga o desinterés**.

---

## Estructura del Repositorio

```
emocionesJSON/
│
├── knnImpl/
│   ├── KNN2.py          # Implementación principal del modelo k-NN
│   ├── KNN2_1.py        # Variante del modelo sin redondeo
│
├── aux/            # Archivos de ayuda
│
├── graficas/      # Scripts y gráficos de análisis exploratorio
│
├── Proyecto Terminal I.pdf   # Documento técnico del proyecto terminal
│
└── README.md            # (Este archivo)
```

---

### Archivos de entrada
Los programas leen datos emocionales desde archivos JSON con la siguiente estructura:

```json
{
  "usuario": "nombre",
  "ronda": 3,
  "valencia": 0.65,
  "arousal": 0.52,
  "riesgo": 0.4,
  "tiempo": 7.2
}
```

Estos valores representan las emociones y condiciones registradas en cada ronda del experimento.

---

## Metodología

El flujo general del sistema consiste en:

1. **Extracción de datos:**  
   Se recopilan registros emocionales (valencia, arousal, tiempo de respuesta, riesgo, etc.) desde archivos JSON.

2. **Entrenamiento del modelo:**  
   Se emplea el algoritmo **k-Nearest Neighbors (k-NN)** para aprender los patrones emocionales del usuario a partir de ejemplos pasados.

3. **Predicción:**  
   Dada una nueva entrada, el modelo predice la respuesta emocional esperada en función de los k vecinos más cercanos correspondiente al msimo algoritmo.

4. **Visualización y evaluación:**  
   Los resultados pueden graficarse para observar la evolución emocional del usuario y la precisión del modelo.

---

## Justificación del uso de k-NN

El algoritmo **k-Nearest Neighbors** fue seleccionado porque:
- No requiere una suposición formal del modelo (ideal para datos subjetivos).  
- Funciona bien con conjuntos pequeños de datos.  
- Aprende directamente de ejemplos previos del mismo usuario.  
- Es simple de interpretar e implementar.  

### Funcionamiento dentro del proyecto:
1. Se guardan las emociones de rondas anteriores.  
2. Para una nueva ronda, el modelo busca los *k* registros más similares.  
3. Promedia sus valores de valencia y arousal.  
4. Predice la emoción esperada para la nueva situación.  

---

## Resultados y conclusiones

Durante la experimentación, el modelo **k-NN** demostró un buen desempeño en la **predicción continua** de emociones, incluso con conjuntos de datos pequeños (en este caso no más de 100).  
Esto permitió automatizar parcialmente la evaluación de nuevos diseños, reduciendo la necesidad de que el usuario respondiera manualmente en cada ronda.

Las conclusiones hasta el momento son:
- La predicción continua **disminuye la carga cognitiva** en tareas repetitivas.  
- k-NN es una alternativa **simple, robusta y eficiente** para datos subjetivos como las emociones.  

---

## Autor

**Isaí Obed Zurita Prado**  
Ingeniería en Computación, UAM Cuajimalpa
Ciudad de México, 2025  
