# 🧠 Predicción de Emociones mediante Aprendizaje Automático
**Proyecto Terminal I – Ingeniería en Computación**  
**Autor:** Isaí Obed Zurita Prado  
**Asesores:** Dra. Alicia Montserrat Alvarado González y Dr. Antonio López Jaimes  
**Año:** 2025  
**Repositorio:** [isaizurita/emocionesJSON](https://github.com/isaizurita/emocionesJSON)

---

## 📘 Descripción General

Este proyecto explora el uso de **aprendizaje automático** para **predecir emociones humanas** en un entorno simulado de toma de decisiones.  
El sistema se basa en los registros emocionales de los usuarios —medidos en las dimensiones de **valencia** (agrado/desagrado) y **arousal** (nivel de activación)— para anticipar sus respuestas frente a nuevos estímulos.

El objetivo principal es **disminuir la carga cognitiva** del usuario y hacer más natural la interacción humano-máquina, especialmente en contextos donde la repetición de decisiones puede generar **fatiga o desinterés**.

---

## 🧩 Estructura del Repositorio

```
emocionesJSON/
│
├── knnImpl/
│   ├── KNN2.py          # Implementación principal del modelo k-NN
│   ├── KNN2_1.py        # Variante del modelo con pruebas y ajustes adicionales
│
├── datasets/            # Archivos JSON con registros de emociones de usuarios
│
├── visualizations/      # Scripts y gráficos de análisis exploratorio (si aplica)
│
├── descripcionpt2.pdf   # Documento técnico del proyecto terminal
│
└── README.md            # (Este archivo)
```

---

## ⚙️ Instalación y Uso

### 🔸 Requisitos previos
Asegúrate de tener instalado:
- Python 3.8 o superior  
- Bibliotecas necesarias:
  ```bash
  pip install numpy pandas scikit-learn matplotlib
  ```

### 🔸 Ejecución básica
1. Clona el repositorio:
   ```bash
   git clone https://github.com/isaizurita/emocionesJSON.git
   cd emocionesJSON/knnImpl
   ```

2. Ejecuta el modelo principal:
   ```bash
   python3 KNN2.py
   ```

3. O bien, usa la versión extendida para pruebas:
   ```bash
   python3 KNN2_1.py
   ```

### 🔸 Archivos de entrada
Los programas leen datos emocionales desde archivos JSON con la siguiente estructura:

```json
{
  "usuario": "u1",
  "ronda": 3,
  "valencia": 0.65,
  "arousal": 0.52,
  "riesgo": 0.4,
  "tiempo": 7.2
}
```

Estos valores representan las emociones y condiciones registradas en cada ronda del experimento.

---

## 🧮 Metodología

El flujo general del sistema consiste en:

1. **Extracción de datos:**  
   Se recopilan registros emocionales (valencia, arousal, tiempo de respuesta, riesgo, etc.) desde archivos JSON.

2. **Limpieza y organización:**  
   Los scripts procesan los datos para eliminar inconsistencias y normalizarlos.

3. **Entrenamiento del modelo:**  
   Se emplea el algoritmo **k-Nearest Neighbors (k-NN)** para aprender los patrones emocionales del usuario a partir de ejemplos pasados.

4. **Predicción:**  
   Dado un nuevo estímulo, el modelo predice la respuesta emocional esperada en función de los k vecinos más cercanos.

5. **Visualización y evaluación:**  
   Los resultados pueden graficarse para observar la evolución emocional del usuario y la precisión del modelo.

---

## 🤖 Justificación del uso de k-NN

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

## 📊 Resultados y conclusiones

Durante la experimentación, el modelo **k-NN** demostró un buen desempeño en la **predicción continua** de emociones, incluso con conjuntos de datos reducidos.  
Esto permitió automatizar parcialmente la evaluación de nuevos diseños, reduciendo la necesidad de que el usuario respondiera manualmente en cada ronda.

Las principales conclusiones son:
- Es posible **predecir emociones humanas** de forma confiable a partir de registros previos.  
- La predicción continua **disminuye la carga cognitiva** en tareas repetitivas.  
- k-NN es una alternativa **simple, robusta y eficiente** para datos subjetivos como las emociones.  
- La metodología puede extenderse a **aplicaciones en educación, salud o entretenimiento**, donde el estado emocional del usuario influye en la interacción.

---

## 🔬 Referencias destacadas

- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.  
- T. Cover and P. Hart, *Nearest Neighbor Pattern Classification*, IEEE Trans. Inf. Theory, 1967.  
- R. Cittadini et al., *Affective state estimation based on Russell’s model and physiological measurements*, *Scientific Reports*, 2023.  
- I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016.  

---

## 🧭 Próximos pasos

- Implementar técnicas de **validación cruzada** para mejorar la robustez del modelo.  
- Explorar modelos más complejos (SVM, Random Forest, Redes Neuronales).  
- Integrar una interfaz visual para mostrar la evolución emocional en tiempo real.  
- Ampliar la base de datos con más usuarios y condiciones experimentales.

---

## 🧑‍💻 Autor

**Isaí Obed Zurita Prado**  
Departamento de Matemáticas Aplicadas y Sistemas  
Ciudad de México, 2025  
