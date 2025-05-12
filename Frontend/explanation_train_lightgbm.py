import streamlit as st

st.set_page_config(page_title="Explicación: train_company_classification.py", layout="wide")

st.title("Análisis del Script: `train_company_classification.py`")

st.markdown("""
Este script no es usado directamente por el chatbot ni se incluye en el actual repositorio pero se deja su explicación porque el modelo LightGBM se mantiene para esta versión final como algoritmo predictivo del nivel económico según los datos recolectados en el chat.
""")
st.caption("Se puede ver su archivo en el repositorio: https://github.com/SebasGalindo/PLN-SegundoMomento")

st.header("Propósito General para el Chatbot")
st.markdown("""
Este script es **crucial** pero se ejecuta **antes** de que el chatbot funcione con los usuarios. Su objetivo principal es **entrenar y guardar el modelo de Machine Learning (LightGBM)** que el chatbot utilizará al final de la conversación para **clasificar el :orange[Nivel Económico]** de la empresa, basándose en los datos recopilados.

En resumen, este script:
1.  Carga datos de empresas ya clasificadas (generados por `generate_dataset.py`).
2.  Prepara estos datos para el entrenamiento.
3.  Busca los mejores parámetros para el modelo LightGBM y lo entrena.
4.  **Guarda el modelo entrenado** (junto con información necesaria como el codificador de etiquetas y las características usadas) en un archivo (`model_bundle_nivel_economico.joblib`). Este archivo es el que **cargará el chatbot (`chatbot_logic.py`)** para hacer la predicción final.
5.  Opcionalmente, genera datos nuevos, aplica el modelo entrenado y las reglas originales para comparar su rendimiento y visualizar los resultados (esto es para evaluar qué tan bien funciona el modelo antes de usarlo en producción).
""")
st.info("Nota: El chatbot en sí mismo *no* ejecuta este script. Solo carga el *resultado* de este script: el archivo `.joblib` con el modelo entrenado.", icon="ℹ️")


st.header("Desglose del Código")

st.subheader("1. Importaciones")
st.code("""
import pandas as pd                 # Para manejar datos en DataFrames
import numpy as np                  # Para operaciones numéricas
import os                           # Para interactuar con el sistema operativo (rutas, directorios)
import sys                          # Para interactuar con el sistema (ej. sys.exit)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold # Para optimizar y validar el modelo
from sklearn.preprocessing import LabelEncoder # Para convertir etiquetas de texto a números
import lightgbm as lgb              # La librería del modelo de clasificación (LightGBM)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # Para evaluar el modelo
import matplotlib.pyplot as plt     # Para crear gráficos
import seaborn as sns               # Para mejorar la visualización de gráficos
import joblib                       # Para guardar y cargar el modelo entrenado
""", language="python")
st.markdown("Se importan librerías estándar para manipulación de datos, machine learning, optimización, evaluación y visualización. `joblib` es clave para guardar el modelo final.")

st.subheader("2. Importación de Funciones Personalizadas")
st.code("""
try:
    # Intenta importar funciones desde otro script local
    from generate_dataset import leer_categorias, generar_dataset_empresas, categorizar_empresas
    print("Funciones importadas exitosamente desde 'generate_dataset.py'")
except ImportError:
    # Si falla, informa al usuario y detiene la ejecución
    print("ERROR: Asegúrate de que el archivo 'generate_dataset.py' está en la misma carpeta.")
    sys.exit(1)
""", language="python")
st.markdown("""
Este bloque intenta importar funciones (`leer_categorias`, `generar_dataset_empresas`, `categorizar_empresas`) definidas en otro archivo llamado `generate_dataset.py`. Estas funciones probablemente se usan para:
* Leer las categorías base de empresas.
* Generar datos sintéticos de empresas (útil para crear datos de prueba).
* Aplicar las reglas de negocio originales para clasificar empresas (para comparar con el modelo).

Si el archivo `generate_dataset.py` no se encuentra, el script se detiene porque depende de él.
""")

st.subheader("3. Configuración")
st.code("""
# --- Configuración ---
CSV_TRAINING_DATA = 'Data/empresas_categorizadas.csv' # Archivo con los datos para entrenar
CATEGORIAS_TXT = 'Data/Categorias-Empresa.txt'        # Archivo con las categorías y sectores base
NUM_NEW_EMPRESAS = 1500                              # Número de empresas nuevas a generar para pruebas
BASE_FEATURES = [                                    # Columnas que usará el modelo como entrada
    'Sector', 'Area', 'Numero Empleados', 'Activos (COP)',
    'Cartera (COP)', 'Deudas (COP)'
]
TARGET_COLUMN = 'Nivel Economico'                    # Columna que el modelo debe predecir

BUNDLE_PATH = 'Data/model_bundle_nivel_economico.joblib' # Ruta donde se guardará el modelo final
""", language="python")
st.markdown("""
Aquí se definen constantes importantes:
* Rutas a los archivos de datos (`.csv`, `.txt`).
* El número de empresas de prueba a generar si se necesita.
* `BASE_FEATURES`: La lista **exacta** de columnas (características) que el modelo usará para aprender. El orden y los nombres son importantes.
* `TARGET_COLUMN`: La columna (`Nivel Economico`) que contiene la clasificación que el modelo debe aprender a predecir.
* `BUNDLE_PATH`: La ruta y nombre del archivo donde se guardará el modelo entrenado, el codificador de etiquetas y la lista de features. **Este es el archivo que usará el chatbot.**
""")

st.subheader("4. Funciones Auxiliares Definidas en el Script")
st.markdown("""
El script define varias funciones para organizar el trabajo:
* `cargar_datos_entrenamiento(ruta_csv)`: Carga el archivo CSV especificado en un DataFrame de pandas, manejando errores si no existe o está vacío.
* `preparar_datos_entrenamiento(df, features, target)`:
    * Selecciona las columnas de `features` (X) y la columna `target` (y).
    * Convierte la columna `target` (que tiene texto como 'Nivel Alto', 'Medio', etc.) en números usando `LabelEncoder`, ya que los modelos trabajan con números. Guarda este `LabelEncoder` porque se necesitará después para convertir las predicciones numéricas de nuevo a texto.
    * Convierte las columnas de texto en `features` (como 'Sector', 'Area') a un tipo de dato especial 'category', que LightGBM puede manejar eficientemente.
    * Devuelve los datos listos (X, y_encoded) y el `LabelEncoder` (le).
* `buscar_mejores_parametros(X_train, y_train, categorical_features)`:
    * Utiliza `RandomizedSearchCV` para probar **automáticamente** diferentes combinaciones de configuraciones (hiperparámetros) para el modelo LightGBM.
    * El objetivo es encontrar la combinación que dé la **mejor precisión (accuracy)** usando validación cruzada (`StratifiedKFold`) para evitar sobreajuste.
    * Devuelve el **mejor modelo encontrado** ya configurado con esos parámetros óptimos.
* `predecir_nuevos_datos(model, label_encoder, df_nuevos, features)`:
    * Toma un modelo ya entrenado (`model`), el `label_encoder` correspondiente, y un DataFrame con nuevos datos (`df_nuevos`).
    * Prepara los nuevos datos (asegurándose de que las columnas categóricas tengan el tipo correcto).
    * Usa `model.predict()` para obtener las predicciones (que serán números).
    * Usa `label_encoder.inverse_transform()` para convertir esas predicciones numéricas de nuevo a las etiquetas de texto originales ('Nivel Alto', 'Medio', etc.).
    * Devuelve las predicciones en formato de texto.
* `graficar_matriz_confusion(y_true, y_pred, labels, title)`:
    * Genera y muestra un gráfico (matriz de confusión) que compara las predicciones del modelo (`y_pred`) con los valores reales (`y_true`). Es útil para ver visualmente qué tipo de errores comete el modelo.
* `cargar_y_usar_modelo_guardado(df_nuevos_para_predecir, features, bundle_path)`:
    * Carga el archivo `.joblib` (el "bundle") que contiene el modelo, el encoder y las features guardadas previamente.
    * Extrae estos componentes.
    * Utiliza la función `predecir_nuevos_datos` para hacer predicciones en `df_nuevos_para_predecir` usando el modelo cargado.
    * Devuelve las predicciones. (Esta función es un ejemplo de cómo el *chatbot* cargaría y usaría el modelo).
""")

st.subheader("5. Bloque Principal de Ejecución")
st.markdown("""
Realiza el proceso completo de entrenamiento y evaluación:
""")
st.markdown("1.  **Carga Datos:** Llama a :orange[`cargar_datos_entrenamiento`] para leer `empresas_categorizadas.csv`.")
st.markdown("2.  **Prepara Datos:** Llama a `preparar_datos_entrenamiento` para obtener `X_train`, `y_train_encoded` y `le`.")
st.markdown("3.  **Entrena (con Optimización):** Llama a `buscar_mejores_parametros` para encontrar la mejor configuración de LightGBM y obtener el `modelo_optimizado`.")
st.markdown("""4.  **Guarda el Modelo (¡Clave!):**
    * Crea un diccionario `model_bundle` que contiene el `modelo_optimizado`, el `label_encoder` (`le`) y la lista `BASE_FEATURES`. Es importante guardar las features para asegurar que el chatbot use las mismas columnas al predecir.
    * Usa `joblib.dump()` para guardar este diccionario en el archivo especificado por `BUNDLE_PATH` (`Data/model_bundle_nivel_economico.joblib`). **Este archivo es el producto final principal de este script para el chatbot.**""")
st.markdown("5.  **Genera Datos de Prueba:** Llama a `generar_dataset_empresas` (importada de `generate_dataset.py`) para crear un nuevo conjunto de datos (`df_nuevos_generados`) que el modelo no ha visto antes.")
st.markdown("6.  **Predice con Modelo:** Usa `predecir_nuevos_datos` con el `modelo_optimizado` para clasificar las nuevas empresas generadas.")
st.markdown("7.  **Aplica Reglas:** Usa `categorizar_empresas` (importada) para clasificar las *mismas* nuevas empresas usando las reglas originales (probablemente basadas en umbrales definidos en `generate_dataset.py`).")
st.markdown("8.  **Compara Resultados:** Une los resultados del modelo y de las reglas en un solo DataFrame (`df_comparacion`) y calcula métricas como `accuracy` y el `classification_report` para ver qué tan bien se alinea el modelo con las reglas originales en datos nuevos.")
st.markdown("9.  **Visualiza:** Llama a `graficar_matriz_confusion` para mostrar la matriz de confusión de la comparación anterior.")
st.markdown("10. **Demostración de Carga:** Llama a `cargar_y_usar_modelo_guardado` para mostrar cómo se cargaría y usaría el archivo `.joblib` guardado en el paso 4 para predecir en unas pocas muestras nuevas. Esto simula lo que haría el chatbot.")

st.header("Conclusión")
st.success("""
**El script `train_company_classification.py` es el responsable de crear y guardar el "cerebro" de clasificación (el modelo LightGBM) que el chatbot financiero utiliza para determinar el 'Nivel Económico' de una empresa una vez que ha recopilado toda la información necesaria del usuario. Su correcta ejecución y la generación del archivo `model_bundle_nivel_economico.joblib` son fundamentales para la funcionalidad final del chatbot.**
""")

st.write("El resultado usado para el chatbot se entrenó con 15 mil datos de empresas y generó los siguientes resultados:")
st.image("Data/matriz_confusion_lightgbm.png", caption="Resultados del entrenamiento del modelo LightGBM para clasificar el 'Nivel Económico' de empresas expresado en la matriz de confusión.", use_container_width=True)