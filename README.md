# Chatbot Financiero con IA (UdeC)
### Integrantes:
**John Sebastian Galindo Hernandez**
**Miguel Angel Moreno Beltran**


Este proyecto presenta un chatbot diseñado para realizar un análisis preliminar de la salud financiera de una empresa. El usuario interactúa con el chatbot proporcionando datos clave, y el sistema utiliza un modelo de Lenguaje de Gran Escala (LLM) a través de la API de Gemini de Google para gestionar la conversación y extraer la información. Posteriormente, un modelo de Machine Learning (LightGBM) clasifica el nivel económico de la empresa y se ofrece una retroalimentación al usuario.

## Objetivo del Proyecto

Refactorizar y mejorar un chatbot financiero existente, reemplazando los modelos BERT de Procesamiento de Lenguaje Natural (PLN) por la API de Gemini para la interacción con el usuario y la extracción de datos. El objetivo es lograr una conversación más natural y una extracción de información más flexible, manteniendo la capacidad de clasificar el nivel económico de una empresa y ofrecer recomendaciones.

## Requisitos de la Solución

El chatbot debe ser capaz de:

1.  Mantener una conversación fluida y amigable con el usuario.
2.  Extraer la siguiente información clave sobre la empresa del usuario:
    * Nombre de la empresa (`nombre_empresa`)
    * Área o categoría principal de actividad (`area_categoria`)
    * Número total de empleados (`numero_empleados`)
    * Valor de los ingresos anuales O el valor total de los activos (`ingresos_o_activos`)
    * Valor de la cartera o cuentas por cobrar (`valor_cartera`)
    * Valor de las deudas totales (`valor_deudas`)
3.  Intentar extraer el nombre del usuario (`nombre_usuario`) de forma opcional.
4.  Inferir el sector económico (`sector`: Primario, Secundario, Terciario, Cuaternario) basado en el `area_categoria` proporcionada.
5.  Gestionar la confirmación de los datos extraídos con el usuario.
6.  Una vez recolectada y confirmada toda la información, utilizar un modelo LightGBM pre-entrenado para clasificar el nivel económico de la empresa.
7.  Presentar al usuario la clasificación obtenida junto con consejos generales adaptados a dicha clasificación.
8.  Ofrecer la opción de iniciar un nuevo análisis para otra empresa.

## Arquitectura de la Solución

La solución se divide en dos componentes principales:

* **Backend (`Backend/chatbot_logic.py`):** Contiene toda la lógica de negocio, incluyendo:
    * Interacción con la API de Gemini de Google.
    * Gestión del estado de la conversación.
    * Conversión de moneda.
    * Ejecución del modelo LightGBM para la predicción.
    * Generación de mensajes de consejo.
* **Frontend (`chatbot_streamlit.py`):** Interfaz de usuario web construida con Streamlit, que permite al usuario interactuar con el chatbot.

### Flujo General

1.  El usuario accede a la aplicación web (Frontend).
2.  El Frontend inicializa una sesión de chat con el Backend.
3.  El Backend crea una sesión de chat con la API de Gemini, configurada con instrucciones de sistema y un esquema de respuesta JSON.
4.  El usuario envía mensajes a través del Frontend.
5.  El Backend reenvía estos mensajes a la API de Gemini.
6.  Gemini procesa el mensaje del usuario, extrae información, gestiona el estado de la conversación (campos pendientes, completados, etc.) y devuelve una respuesta JSON estructurada al Backend.
7.  El Backend recibe el JSON, lo parsea a un diccionario Python y lo reenvía (o la parte relevante) al Frontend.
8.  El Frontend muestra el mensaje del chatbot (`message_for_user`) al usuario y actualiza su estado interno.
9.  Este ciclo se repite hasta que Gemini indica que todos los datos necesarios han sido recolectados (`interaction_type: "datos_completos"`).
10. El Backend toma los datos completados del estado de la conversación, los prepara y los pasa al modelo LightGBM.
11. El modelo LightGBM predice la categoría del nivel económico.
12. El Backend genera un mensaje final con la clasificación y consejos.
13. El Frontend muestra este análisis final al usuario.

## Tecnologías Utilizadas

* **Python**: Lenguaje de programación principal.
* **Google Gemini API**: Para el procesamiento del lenguaje natural, gestión de diálogo y extracción de información. Se utiliza a través de la biblioteca `google-genai`.
* **Streamlit**: Para la creación de la interfaz de usuario web.
* **Pydantic**: Para definir el esquema de respuesta esperado de la API de Gemini y asegurar la estructura de los datos.
* **LightGBM**: Modelo de Machine Learning para la clasificación del nivel económico.
* **Joblib**: Para cargar el modelo LightGBM pre-entrenado.
* **Pandas**: Para la manipulación de datos, especialmente para preparar la entrada al modelo LightGBM.
* **yfinance**: (Opcional, para obtener tasas de cambio actualizadas).

## Configuración y Ejecución

### Prerrequisitos

* Python 3.9 o superior.
* Tener una API Key de Google Gemini.

### Instalación

1.  Clonar el repositorio:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```
2.  (Recomendado) Crear y activar un entorno virtual:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Asegúrate de tener un archivo `requirements.txt` con todas las bibliotecas necesarias, como `streamlit`, `google-genai`, `pydantic`, `joblib`, `pandas`, `lightgbm`, `yfinance`)*

### Configuración

1.  **API Key de Gemini**:
    * Crea un archivo llamado `credentials.json` dentro de la carpeta `Data/`.
    * El contenido del archivo debe ser un JSON con tu API Key:
        ```json
        {
            "API_KEY": "TU_API_KEY_DE_GEMINI_AQUI"
        }
        ```
2.  **Modelo LightGBM**:
    * Asegúrate de que tu modelo LightGBM entrenado (`model_bundle_nivel_economico.joblib`) esté presente en la carpeta `Data/`.
3.  **Instrucciones del Sistema para Gemini**:
    * El archivo `Data/base_chatbot_instructions.txt` contiene el prompt del sistema que guía el comportamiento de Gemini. Puedes modificarlo según tus necesidades.

### Ejecución

Para iniciar la aplicación Streamlit:

```bash
streamlit run ./Frontend/streamlit_app.py
```

### Ver en la WEB
El chatbot se puede ver en la web en el siguiente enlace: https://chatbot-financiero-udec-sgmm.streamlit.app/
