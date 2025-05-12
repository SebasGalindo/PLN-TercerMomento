import streamlit as st

st.set_page_config(page_title="Explicación: generate_dataset.py", layout="wide")

st.title("Análisis del Script: `generate_dataset.py`")

st.markdown("""
Este script no es usado directamente por el chatbot ni se incluye en el actual repositorio pero se deja su explicación porque el modelo LightGBM se mantiene para esta versión final como algoritmo predictivo del nivel económico según los datos recolectados en el chat.
""")
st.caption("Se puede ver su archivo en el repositorio: https://github.com/SebasGalindo/PLN-SegundoMomento")

st.header("Propósito General para el Chatbot")

st.markdown("""
Este script cumple una función **preparatoria fundamental** para el sistema del chatbot: **genera el conjunto de datos etiquetado (`empresas_categorizadas.csv`) que se utiliza para entrenar el modelo de clasificación** en `train_company_classification.py`. Su lógica se basa en crear datos sintéticos de empresas que simulen una realidad económica variada y, lo más importante, **aplicar un conjunto de reglas financieras predefinidas para asignar a cada empresa simulada un 'Nivel Económico'**. Este dataset etiquetado es esencial para que el modelo de Machine Learning aprenda a replicar esta clasificación basada en reglas.
""")
st.info("Nota: Este script define las 'reglas del juego' o el 'conocimiento experto' que el modelo de IA intentará aprender. No interactúa directamente con el usuario final.", icon="ℹ️")

st.header("Desglose del Código")

st.subheader("1. Importaciones y Lectura de Categorías (`leer_categorias`)")

st.markdown("""
El script comienza importando las librerías necesarias: `pandas` para la manipulación de datos tabulares, `numpy` para operaciones numéricas, `faker` para generar datos ficticios (nombres de empresas) y `os` para manejo de directorios. La función `leer_categorias` se encarga de cargar un archivo de texto (`Categorias-Empresa.txt`) que contiene las posibles combinaciones de 'Area' y 'Sector' empresarial. Este DataFrame de categorías sirve como base para asegurar que las empresas generadas pertenezcan a clasificaciones válidas y diversas.
""")
st.code("""
import pandas as pd
import numpy as np
from faker import Faker
import os

# --- 1. Función para leer el TXT con categorías y sectores (sin cambios) ---
def leer_categorias(ruta_archivo_txt):
    # ... (código para leer y limpiar el archivo TXT) ...
    return df_categorias
""", language="python")

st.subheader("2. Generación de Datos Sintéticos (`generar_dataset_empresas`)")

st.markdown("""
La función `generar_dataset_empresas` crea un DataFrame con datos de empresas ficticias. Utiliza la librería `Faker` para nombres realistas y genera valores numéricos (Número de Empleados, Activos, Cartera, Deudas) de forma aleatoria pero **coherente**. Introduce una lógica importante: **los rangos de generación de estos valores se ajustan según el 'Sector'** al que pertenece la empresa (Primario, Secundario, Terciario, Cuaternario). Por ejemplo, se asume que empresas de sectores Primario/Secundario (más industriales) podrían tener rangos de activos por empleado más altos o tolerar mayores niveles de deuda en comparación con empresas del sector Cuaternario (basadas en conocimiento). Esta dependencia del sector busca crear un dataset más realista y representativo de las diferencias estructurales entre industrias.
""")
st.code("""
# --- 2. Función para generar el dataset aleatorio (MODIFICADA) ---
def generar_dataset_empresas(df_categorias, num_empresas=100):
    # ... (Inicialización de Faker, bucle para generar empresas) ...

        # --- AJUSTES POR SECTOR ---
        if sector in ['Primario', 'Secundario']:
            # Rangos para sectores intensivos en capital
            empleados_min, empleados_max = 10, 2500
            # ... (otros ajustes de rangos)
        elif sector == 'Cuaternario':
            # Rangos para sectores de conocimiento
            # ...
        else: # Terciario y otros
            # Rangos intermedios
            # ...

        # Generar valores aleatorios usando los rangos ajustados
        num_empleados = np.random.randint(empleados_min, empleados_max + 1)
        # ... (generación de activos, deudas, cartera con influencia de rangos) ...

    # ... (retorna el DataFrame con datos generados) ...
    return pd.DataFrame(data_empresas)
""", language="python")

st.subheader("3. Cálculo de Métricas y Clasificación (`categorizar_empresas`)")

st.markdown("""
Esta es la función **central** del script, donde se aplica la lógica financiera para clasificar las empresas generadas. Recibe el DataFrame de empresas y realiza los siguientes pasos clave:

**a) Cálculo de Indicadores Financieros:**
Calcula dos métricas fundamentales a partir de los datos básicos (Activos, Deudas):
""")

st.markdown("* **`Patrimonio Neto (COP)`:** Calculado como `Activos (COP) - Deudas (COP)`.")
st.markdown("""
    * **Importancia y Significado:** El Patrimonio Neto representa el valor contable residual de la empresa después de cubrir todas sus obligaciones; es la porción de los activos que pertenece efectivamente a los propietarios o accionistas. Un **Patrimonio Neto negativo es un indicador crítico de insolvencia**, significando que la empresa debe más de lo que posee. Por ello, es la primera condición evaluada para asignar la categoría `'En Quiebra Técnica / Insolvente'`, reflejando una situación financiera insostenible.
""")
st.code("""
    # Calcular Patrimonio Neto
    df['Patrimonio Neto (COP)'] = df['Activos (COP)'] - df['Deudas (COP)']
""", language="python")


st.markdown("* **`Razon Endeudamiento`:** Calculado como `Deudas (COP) / Activos (COP)` (manejando el caso de Activos cero para evitar división por cero).")
st.markdown("""
    * **Importancia y Significado:** Este ratio mide qué porcentaje de los activos totales de la empresa ha sido financiado mediante deuda. Es un indicador clave del **apalancamiento financiero y del riesgo asociado**. Un ratio elevado (cercano a 1 o más) indica una alta dependencia de la financiación externa, lo que aumenta la vulnerabilidad ante fluctuaciones económicas o dificultades para generar ingresos, ya que una gran parte de las ganancias podría destinarse al pago de deudas. Un ratio bajo sugiere una estructura financiera más conservadora y, en general, menor riesgo.
""")
st.code("""
    # Calcular Razón de Endeudamiento (con manejo de Activos = 0)
    df['Razon Endeudamiento'] = np.where(
        df['Activos (COP)'] > 0,
        df['Deudas (COP)'] / df['Activos (COP)'],
        np.inf # Asigna infinito si activos es 0 y hay deuda
    )
    df['Razon Endeudamiento'].replace([np.inf, -np.inf], np.nan, inplace=True) # Convierte infinitos a NaN
""", language="python")

st.markdown("""
**b) Aplicación de Reglas de Clasificación (con Ajuste por Sector):**
Una vez calculados el Patrimonio Neto y la Razón de Endeudamiento, se aplica una serie de **condiciones basadas en umbrales** sobre estos indicadores para asignar la etiqueta `Nivel Economico`.

* **Lógica:** Primero se verifica la condición de Patrimonio Neto negativo (insolvencia). Si no se cumple, la clasificación se basa enteramente en la `Razon Endeudamiento`. Se establecen **umbrales progresivos**: cuanto mayor es la razón de endeudamiento, peor es la categoría asignada (desde 'Excelente / Muy Sólida' para ratios muy bajos hasta 'Crítica / Muy Débil' para ratios muy altos).
* **Ajuste por Sector:** Crucialmente, **los umbrales de la Razón de Endeudamiento no son fijos**, sino que se **ajustan dinámicamente según el 'Sector'** de la empresa mediante un `factor_ajuste`. Sectores como Primario/Secundario, que suelen requerir más capital y operar con mayor endeudamiento, tienen umbrales *más altos* (son más tolerantes a la deuda), mientras que sectores como el Cuaternario podrían tener umbrales *más bajos* (se espera menor endeudamiento). Esto hace que la clasificación sea más equitativa y adaptada a las características inherentes de cada industria.

Las categorías resultantes (`'En Quiebra Técnica / Insolvente'`, `'Crítica / Muy Débil'`, `'Vulnerable / Débil'`, `'Estable / Regular'`, `'Sólida / Buena'`, `'Excelente / Muy Sólida'`) reflejan directamente el **nivel de riesgo y solidez financiera** inferido a partir del patrimonio y, principalmente, del nivel de endeudamiento relativo a los activos, considerando las particularidades sectoriales.
""")
st.code("""
    # --- AJUSTES POR SECTOR EN UMBRALES ---
    # ... (cálculo de factor_ajuste basado en Sector) ...
    # Aplicar ajuste a los umbrales base
    umbral_critica_ajustado = umbral_base['Critica'] * factor_ajuste
    # ... (ajuste de otros umbrales) ...

    # Definir condiciones y categorías usando umbrales ajustados
    conditions = [
        (df['Patrimonio Neto (COP)'] < 0) | (df['Razon Endeudamiento'].isna() & (df['Deudas (COP)'] > 0)), # Insolvente
        (df['Razon Endeudamiento'] > umbral_critica_ajustado),           # Crítica
        (df['Razon Endeudamiento'] > umbral_vulnerable_ajustado),        # Vulnerable
        # ... (resto de condiciones basadas en umbrales ajustados) ...
    ]
    categories = [ # Etiquetas correspondientes a cada condición
        'En Quiebra Técnica / Insolvente',
        'Crítica / Muy Débil',
        # ... (resto de etiquetas) ...
    ]
    # Asignar la categoría según la primera condición que se cumpla
    df['Nivel Economico'] = np.select(conditions, categories, default='Indeterminado')

    return df
""", language="python")


st.subheader("4. Guardado del Dataset (`guardar_csv`) y Ejecución Principal")

st.markdown("""
La función `guardar_csv` simplemente toma el DataFrame final (con los datos generados, las métricas calculadas y el 'Nivel Economico' asignado) y lo guarda en un archivo CSV (`empresas_categorizadas.csv` por defecto).
""")

