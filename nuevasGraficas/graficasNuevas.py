import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Ruta de los JSON
carpeta_json = "/Users/isaizurita/emocionesJSON/JSONS4"

# Buscar archivos
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + glob.glob(os.path.join(carpeta_json, "*.JSON"))
print(f"\nSe encontraron {len(archivos_json)} archivos JSON.")

# Carpeta de salida
output_folder = os.path.join(carpeta_json, "graficas4")
os.makedirs(output_folder, exist_ok=True)

df_general = pd.DataFrame()
usuario_id = 0

# Leer archivos
for archivo in archivos_json:
    try:
        with open(archivo, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        profile = data.get("profile", {})
        level = data.get("level", {})

        usuario = (
            profile.get("initials") or
            f"user_{usuario_id}"
        )

        age = profile.get("age")
        gender = profile.get("gender")
        education = profile.get("educationLevel")

        nivelFamiliaridad = level.get("familiarityLevel")
        nivelConocimiento = level.get("knowledgeLevel")

        rondas = data.get("History_Experiment_Simulated", [])

        for ronda_idx, ronda in enumerate(rondas):
            df_ronda = pd.DataFrame(ronda)

            df_ronda['sliderDissatisfiedSatisfied'] = pd.to_numeric(
                df_ronda.get('sliderDissatisfiedSatisfied'), errors='coerce'
            )
            df_ronda['sliderBoredExcited'] = pd.to_numeric(
                df_ronda.get('sliderBoredExcited'), errors='coerce'
            )

            df_ronda['usuario'] = usuario
            df_ronda['ronda'] = ronda_idx
            df_ronda['iteracion'] = range(len(df_ronda))
            df_ronda['global_id'] = usuario_id

            df_ronda['age'] = age
            df_ronda['gender'] = gender
            df_ronda['education'] = education
            df_ronda['nivelFamiliaridad'] = nivelFamiliaridad
            df_ronda['nivelConocimiento'] = nivelConocimiento

            df_general = pd.concat([
                df_general,
                df_ronda[['usuario', 'ronda', 'iteracion',
                          'sliderDissatisfiedSatisfied', 'sliderBoredExcited',
                          'time', 'risk', 'arrival',
                          'age', 'gender', 'education',
                          'nivelFamiliaridad', 'nivelConocimiento',
                          'global_id']]
            ], ignore_index=True)

        usuario_id += 1

    except Exception as e:
        print(f"Error en archivo {archivo}: {e}")
        continue

print("\nUsuarios encontrados:")
print(df_general['usuario'].unique())

usuarios = df_general['usuario'].unique()
colors = plt.get_cmap('tab10')

for user in usuarios:
    df_user = df_general[df_general['usuario'] == user]
    plt.figure(figsize=(10, 5))

    markers = itertools.cycle(('o', 's', 'D', '^', 'v', '<', '>', 'x', '*', '+'))

    for i, ronda in enumerate(sorted(df_user['ronda'].unique())):
        df_r = df_user[df_user['ronda'] == ronda]
        marker = next(markers)
        color = colors(i % 10)

        plt.plot(df_r['iteracion'], df_r['sliderDissatisfiedSatisfied'],
                 label=f'R{ronda} - Valence', color=color, marker=marker)

        plt.plot(df_r['iteracion'], df_r['sliderBoredExcited'],
                 label=f'R{ronda} - Arousal', color=color,
                 marker=marker, linestyle='dashed')

    edad = df_user['age'].iloc[0]
    conocimiento = df_user['nivelConocimiento'].iloc[0]
    familiaridad = df_user['nivelFamiliaridad'].iloc[0]

    plt.title(f'Usuario {user} | Edad: {edad} | Conocimiento: {conocimiento}')
    plt.xlabel("Iteración")
    plt.ylabel("Valor del Slider")
    plt.legend()
    plt.grid(True)

    max_iter = df_user['iteracion'].max()
    plt.xticks(range(0, max_iter + 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{user}_rondas.png"))
    plt.close()

# Promedios por usuario
df_promedio = df_general.groupby('usuario').agg({
    'sliderDissatisfiedSatisfied': 'mean',
    'sliderBoredExcited': 'mean',
    'nivelConocimiento': 'first',
    'nivelFamiliaridad': 'first'
}).reset_index()

#Conocimiento vs emoción 
plt.figure()
plt.scatter(df_promedio['nivelConocimiento'], df_promedio['sliderDissatisfiedSatisfied'])
plt.xlabel("Nivel de conocimiento")
plt.ylabel("Valence promedio")
plt.title("Conocimiento vs emoción (valence)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "conocimiento_vs_valence.png"))
plt.close()

# Familiaridad vs emoción 
plt.figure()
plt.scatter(df_promedio['nivelFamiliaridad'], df_promedio['sliderDissatisfiedSatisfied'])
plt.xlabel("Nivel de familiaridad")
plt.ylabel("Valence promedio")
plt.title("Familiaridad vs emoción (valence)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "familiaridad_vs_valence.png"))
plt.close()

csv_path = os.path.join(carpeta_json, "rondas_usuarios.csv")
json_path = os.path.join(carpeta_json, "rondas_usuarios.json")

df_general.to_csv(csv_path, index=False)

df_general.drop(columns=['time', 'risk', 'arrival']).to_json(
    json_path, orient="records", indent=2
)

print("\nProceso finalizado.")
print(f"CSV: {csv_path}")
print(f"JSON: {json_path}")
print(f"Gráficas: {output_folder}")