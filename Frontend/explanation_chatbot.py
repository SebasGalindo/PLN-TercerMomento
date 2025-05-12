import streamlit as st

st.title("Explicación Detallada del Chatbot Financiero UdeC 🤖💡")
st.markdown("""
¡Bienvenido a la explicación técnica de nuestro Chatbot Financiero!
Este proyecto ha sido refactorizado para utilizar la potencia de los Modelos de Lenguaje de Gran Escala (LLM)
a través de la API de Gemini de Google, combinada con un modelo de Machine Learning (LightGBM)
para ofrecer un análisis preliminar de la salud financiera de una empresa.
""")

st.header("🎯 Objetivo del Proyecto")
st.markdown("""
El objetivo principal es permitir a un usuario ingresar información financiera clave de su empresa
a través de una interfaz conversacional amigable. El chatbot guía al usuario, recolecta los datos
necesarios, y finalmente presenta una clasificación del nivel económico de la empresa junto con
recomendaciones generales.

Esta versión reemplaza los modelos BERT utilizados anteriormente por la API de Gemini para
lograr una interacción más natural y una extracción de información más flexible y robusta.
""")

st.header("📋 Requisitos Funcionales Clave")
st.markdown("""
El chatbot debe ser capaz de:
* Mantener una conversación fluida y amigable.
* Extraer datos como: nombre de la empresa, área de actividad, número de empleados, ingresos/activos, cartera y deudas.
* Intentar obtener el nombre del usuario.
* Inferir el sector económico (Primario, Secundario, Terciario, Cuaternario) basado en el área de actividad.
* Confirmar los datos extraídos con el usuario.
* Utilizar un modelo LightGBM pre-entrenado para clasificar el nivel económico.
* Presentar la clasificación y consejos adaptados.
* Permitir iniciar un nuevo análisis.
""")

st.header("🛠️ Arquitectura y Flujo de Trabajo")
st.markdown("""
La solución se divide en un **Frontend** (interfaz de usuario con Streamlit) y un **Backend** (lógica del chatbot).

**Flujo General:**
1.  El usuario interactúa con el Frontend (`chatbot_streamlit.py`).
2.  El Frontend inicializa una sesión de chat con el Backend (`Backend/chatbot_logic.py`).
3.  El Backend crea y configura una sesión de chat con la API de Gemini (`google.genai`), especificando:
    * Una **instrucción de sistema** detallada (desde `Data/base_chatbot_instructions.txt`) que guía el comportamiento, tono y formato de respuesta de Gemini.
    * Un **esquema de respuesta JSON** (definido con clases Pydantic) para asegurar que Gemini devuelva la información de manera estructurada.
4.  Cuando el usuario envía un mensaje, el Frontend lo pasa al Backend.
5.  El Backend envía el mensaje a la API de Gemini a través de la sesión de chat activa.
6.  Gemini procesa el mensaje, considerando el historial de la conversación, la instrucción del sistema y el esquema de respuesta. Devuelve un string JSON.
7.  El Backend recibe este string, lo limpia (si es necesario, ej: quita ```json ```) y lo parsea a un diccionario Python usando `json.loads()`.
8.  El Backend devuelve este diccionario al Frontend.
9.  El Frontend utiliza la información del diccionario (ej: `message_for_user`, `conversation_state`) para actualizar la interfaz y mostrar la respuesta del chatbot.
10. El ciclo se repite. El `conversation_state` devuelto por Gemini se utiliza para rastrear los campos pendientes y completados.
11. Cuando Gemini indica que todos los datos están completos (`interaction_type: "datos_completos"`), el Backend toma el `conversation_state`, prepara los datos (incluyendo conversión a COP e inferencia de sector si es necesario) y los pasa al modelo LightGBM.
12. El modelo LightGBM predice la categoría del nivel económico.
13. El Backend genera un mensaje final con esta clasificación y consejos relevantes.
14. El Frontend muestra este análisis final.
""")

with st.expander("⚙️ Profundizando en el Backend (`Backend/chatbot_logic.py`)", expanded=False):
    st.subheader("1. Inicialización y Dependencias (`_initialize_client_and_dependencies`)")
    st.markdown("""
    Esta función se ejecuta una vez al cargar el módulo y se encarga de:
    * **Cargar API Key de Gemini**: Lee las credenciales desde `Data/credentials.json`.
        * **Crear Cliente `google.genai.Client`**: Instancia el cliente para interactuar con la API de Gemini.
            ```python
            # En _initialize_client_and_dependencies()
            with open(CREDENTIALS_PATH, 'r') as f: credentials = json.load(f)
            API_KEY = credentials.get("API_KEY")
            genai_client_instance = genai.Client(api_key=API_KEY)
            ```
        * **Cargar Instrucción del Sistema**: Lee el prompt principal desde `Data/base_chatbot_instructions.txt`. Este archivo es VITAL, ya que define la personalidad, el flujo, los datos a recolectar y las reglas de formato JSON para Gemini.
        * **Cargar Modelo LightGBM**: Usa `joblib` para cargar el modelo de clasificación (`.joblib`), el codificador de etiquetas y la lista de características esperadas.
        * **Obtener Tasas de Cambio**: Intenta obtener tasas de cambio actualizadas (USD/COP, EUR/COP) usando `yfinance`. Si falla, utiliza valores por defecto.
        """)

    st.subheader("2. Definición del Esquema de Respuesta con Pydantic")
    st.markdown("""
    Se definen varias clases Pydantic (`UpdatedField`, `CompletedField`, `ConversationState`, `ChatbotResponse`) que describen la estructura exacta del JSON que esperamos que la API de Gemini devuelva.
    La clase principal es `ChatbotResponse`. Estas clases **no se usan para instanciar objetos en Python después de recibir el JSON del API**, sino que **se pasan a la configuración de la API de Gemini como `response_schema`**.
    Esto instruye a Gemini para que formatee su salida de acuerdo a este esquema.

        ```python
        # Ejemplo de la clase principal para el response_schema
        class ChatbotResponse(BaseModel):
            interaction_type: str
            message_for_user: str
            updated_field: UpdatedField = Field(default_factory=UpdatedField)
            next_question_key: Optional[str] = Field(None)
            conversation_state: ConversationState = Field(default_factory=ConversationState)
        ```
        """)

    st.subheader("3. Creación de la Sesión de Chat con Gemini (`create_new_chat`)")
    st.markdown("""
    Esta función es llamada por el Frontend para iniciar una nueva conversación.
    * Utiliza `genai_client_instance.chats.create(model=model_name, config=...)`.
    * La parte más importante es el parámetro `config`, donde se pasa un objeto `google_genai_types.GenerateContentConfig`. Este objeto contiene:
        * `system_instruction`: El prompt detallado cargado previamente.
            * `response_mime_type="application/json"`: Le dice a Gemini que la respuesta debe ser un JSON.
            * `response_schema=ChatbotResponse`: Le pasa la clase Pydantic para que Gemini formatee el JSON según esa estructura. ¡Esta es la clave para obtener JSONs estructurados y fiables!
        * Devuelve un objeto de sesión de chat que mantiene el historial de la conversación.

        ```python
        # Dentro de create_new_chat()
        chat_session = genai_client_instance.chats.create(
            model="gemini-2.0-flash", # o el modelo especificado
            config=google_genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_FINASSISTENTE,
                response_mime_type="application/json",
                response_schema=ChatbotResponse, # Especifica la estructura de salida
            )
        )
        return chat_session
        ```
        """)

    st.subheader("4. Envío de Mensajes y Procesamiento de Respuestas (`send_message_to_chat`)")
    st.markdown("""
    Gestiona cada turno de la conversación.
    * Recibe la sesión de chat activa y el mensaje del usuario.
    * Llama a `chat_session.send_message(user_message)`.
    * Accede a `api_response.text` para obtener el string JSON devuelto por Gemini (gracias a la configuración del chat).
        * Limpia el string JSON (remueve ```json ``` si están presentes).
        * Parsea el string a un diccionario Python usando `json.loads()`.
        * Realiza una validación mínima para asegurar que el diccionario contiene campos clave esperados.
        * **Devuelve el diccionario Python directamente**, junto con un mensaje de error si algo falló.

        ```python
        # Dentro de send_message_to_chat()
        api_response = chat_session.send_message(user_message)
        # ... limpieza de cleaned_json_text ...
        parsed_dict_response = json.loads(cleaned_json_text)
        # ... validación mínima del dict ...
        return parsed_dict_response, None
        ```
        """)

    st.subheader("5. Lógica de Negocio Auxiliar")
    st.markdown("""
    * **`_convert_to_cop`**: Convierte valores monetarios de USD o EUR a Pesos Colombianos (COP) usando las tasas de cambio. Es crucial para normalizar los datos antes del análisis LightGBM.
    * **`run_lgbm_analysis`**:
        * Se activa cuando Gemini indica que todos los datos necesarios han sido recolectados (`interaction_type: "datos_completos"`).
        * Recibe el `conversation_state` (como diccionario) de la última respuesta de Gemini.
            * Transforma la lista de `completed_fields` en un formato más manejable (un diccionario mapeando nombre de campo a su objeto/valor).
            * **Inferencia de Sector**: Verifica si el campo `sector` fue inferido por Gemini (según la instrucción del sistema). Si no (o es "Desconocido"), realiza una llamada "one-shot" a `genai_client_instance.models.generate_content(...)` con un prompt específico para clasificar el `area_categoria` en Primario, Secundario, Terciario o Cuaternario.
            * Prepara un DataFrame de Pandas con los datos numéricos (convertidos a COP) y categóricos.
            * Asegura que las columnas y tipos de datos coincidan con lo que el modelo LightGBM espera.
            * Utiliza el `lgbm_model` cargado para predecir el nivel económico.
            * Devuelve la etiqueta de la categoría predicha (ej: "Estable / Regular", "Crítica / Muy Débil").
        * **`generate_final_message_with_advice`**:
            * Toma la categoría predicha por LightGBM y el nombre de la empresa.
            * Construye un mensaje de resumen formateado en Markdown, presentando la clasificación y ofreciendo una serie de consejos generales y accionables adaptados a cada nivel económico posible.
        """)

with st.expander("🎨 Explorando el Frontend (`chatbot_streamlit.py`)", expanded=False):
    st.subheader("1. Configuración General de la Aplicación Streamlit")
    st.markdown("""
    * **`st.set_page_config`**: Define el título de la pestaña del navegador, el ícono y el layout (centrado o ancho).
    * **`st.title` y `st.caption`**: Muestran el título principal y un subtítulo en la página.
    * **Importación del Backend**: Intenta importar el módulo `Backend.chatbot_logic` como `bot`. Si hay un error, lo muestra y detiene la aplicación.
    """)

    st.subheader("2. Inicialización y Gestión del Estado de la Sesión (`st.session_state`)")
    st.markdown("""
    Streamlit utiliza `st.session_state` para mantener información entre interacciones del usuario y reruns de la página.
    * **`initialize_new_chat_session()`**:
        * Llamada al inicio y cuando se presiona "Nuevo Chat".
        * Invoca a `bot.create_new_chat()` para obtener una nueva `gemini_chat_session` y la guarda en `st.session_state.gemini_chat_session`.
        * Reinicia `st.session_state.chat_history` (una lista para los mensajes).
        * `st.session_state.chatbot_state` (que almacenará el diccionario `conversation_state` de Gemini) se pone a `None` inicialmente.
        * `st.session_state.analysis_done` se pone a `False`.
        * Añade un mensaje de bienvenida fijo al `chat_history` para que el usuario vea algo al iniciar.
        * **Botón "Nuevo Chat"**: Ubicado en la `st.sidebar`, llama a `initialize_new_chat_session()` y luego a `st.rerun()` para refrescar la interfaz.

        ```python
        # Ejemplo de inicialización de nuevo chat en el frontend
        if st.sidebar.button("✨ Nuevo Chat", key="new_chat_button"):
            initialize_new_chat_session()
            st.rerun()

        if 'gemini_chat_session' not in st.session_state:
            initialize_new_chat_session()
        ```
        """)

    st.subheader("3. Interfaz de Chat")
    st.markdown("""
    * **Mostrar Historial (`st.chat_message`, `st.markdown`)**: Itera sobre `st.session_state.chat_history`. Cada mensaje se muestra dentro de un `st.chat_message(role)` (donde `role` es "user" o "assistant") y se renderiza con `st.markdown()` para permitir formato.
    * **Entrada del Usuario (`st.chat_input`)**: Proporciona el campo de texto para que el usuario escriba. Se deshabilita si `st.session_state.analysis_done` es `True`.

    ```python
        # Mostrar mensajes
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input del usuario
        user_prompt = st.chat_input("Escribe tu respuesta aquí...", disabled=user_input_disabled)
        ```
        """)

    st.subheader("4. Flujo de Interacción con el Backend")
    st.markdown("""
    Cuando el usuario ingresa un mensaje (`user_prompt`):
    1.  El mensaje del usuario se añade al `chat_history` y se muestra.
    2.  Se muestra un `st.spinner("FinAsistente está pensando... 🧠")`.
    3.  Se llama a `bot.send_message_to_chat(st.session_state.gemini_chat_session, user_prompt)`.
    4.  Se recibe `response_dict, error_msg` del backend.
    5.  **Manejo de Respuesta/Error**:
        * Si hay `error_msg` (o `response_dict` es `None`), se muestra un mensaje de error en la interfaz.
        * Si no hay error, se extrae `response_dict.get("message_for_user")` y se muestra como la respuesta del chatbot. Este mensaje se añade también al `chat_history`.
        * Se actualiza `st.session_state.chatbot_state = response_dict.get("conversation_state")`.
    6.  **Verificación de Fin de Conversación y Análisis**:
        * Se comprueba si `response_dict.get("interaction_type") == "datos_completos"`.
        * Si es así, se marca `st.session_state.analysis_done = True`.
        * Se muestra otro spinner indicando "Realizando análisis financiero final... 📊".
        * Se llama a `bot.run_lgbm_analysis(st.session_state.chatbot_state)` (pasando el diccionario `conversation_state`).
        * Se obtiene el nombre de la empresa del `chatbot_state` para personalizar.
        * Se llama a `bot.generate_final_message_with_advice()` para obtener el mensaje de análisis.
        * Este mensaje final se muestra con `st.markdown()` y se añade al historial.
        * Se invoca `st.rerun()` para actualizar la UI y deshabilitar el campo de input.
    """)

st.markdown("---")
st.markdown("Este chatbot representa una integración de tecnologías de IA para un propósito práctico, combinando la flexibilidad conversacional de los LLMs con la precisión predictiva de modelos de Machine Learning tradicionales.")
