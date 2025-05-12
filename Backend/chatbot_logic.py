# Backend/chatbot_logic.py
# Este archivo contiene la lógica principal del chatbot.
# Authors: John Sebastián Galindo Hernández y Miguel Ángel Moreno Beltrán

# region 0. Imports
# ES: Importaciones
import json # for charge gemini credentials and load the chat API response
import joblib # for load the lgbm model
import pandas as pd # for data processing, specifically for the prediction with the lgbm model
from pathlib import Path # for file paths
from typing import List, Dict, Any, Optional, Tuple # for type hints
from google import genai # for gemini API
from google.genai import types as google_genai_types
from pydantic import BaseModel, Field # for define the schema for Gemini
import re # for remove multiple spaces
import streamlit as st # for access to secrets in the deployed app
# endregion

#region 1. Pydantic Models for `response_schema` in order to receive a structured JSON response from Gemini
# ES: Modelos Pydantic para `response_schema` de forma que Gemini pueda devolver un JSON estructurado
class UpdatedField(BaseModel):
    """
    EN: Class to represent an updated field in the chatbot response.
    ES: Clase para representar un campo actualizado en la respuesta del chatbot.
    """
    field_name: Optional[str] = Field(None) 
    extracted_value: Optional[str] = Field(None)
    confirmation_value: Optional[str] = Field(None)

class CompletedField(BaseModel):
    """
    EN: Class to represent a completed field in the chatbot response.
    ES: Clase para representar un campo completado en la respuesta del chatbot.
    """
    name: str
    value: str
    currency: Optional[str] = Field(None)

class ConversationState(BaseModel):
    """
    EN: Class to represent the state of the conversation.
    ES: Clase para representar el estado de la conversación.
    """
    pending_fields: List[str] = Field(default_factory=list)
    completed_fields: List[CompletedField] = Field(default_factory=list)
    waiting_for_confirmation: bool = Field(default=False)
    detected_user_name: Optional[str] = Field(None)

class ChatbotResponse(BaseModel):
    """
    EN: Main Class to represent the response from the chatbot.
    ES: Clase principal para representar la respuesta del chatbot.
    """
    interaction_type: str
    message_for_user: str
    updated_field: UpdatedField = Field(default_factory=UpdatedField) 
    next_question_key: Optional[str] = Field(None)
    conversation_state: ConversationState = Field(default_factory=ConversationState) 
    
#endregion

#region # --- 2. Initial Constants and Configuration ---
# ES: Constantes y configuración iniciales
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
CREDENTIALS_PATH = DATA_DIR / "credentials.json"
LGBM_MODEL_PATH = DATA_DIR / "model_bundle_nivel_economico.joblib"
SYSTEM_PROMPT_PATH = DATA_DIR / "base_chatbot_instructions.txt"

REQUIRED_FIELDS_TO_COLLECT = [
    "nombre_empresa", "area_categoria", "numero_empleados",
    "ingresos_o_activos", "valor_cartera", "valor_deudas"
] # sector is inferred, nombre_usuario is optional

API_KEY = None
genai_client_instance = None
lgbm_model = None
lgbm_label_encoder = None
lgbm_features = None
exchange_rates = { 'USD': 4200, 'EUR': 4800, 'COP': 1 }
SYSTEM_INSTRUCTION_FINASSISTENTE = ""
#endregion

#region # --- 3. Client and Model Initialization ---
# ES: Inicialización del cliente y las dependencias (Instrucciones, API Key, Modelo LGBM, Tasas de Cambio)
def _initialize_client_and_dependencies():
    """
    EN: Initializes the client and dependencies (Instructions, API Key, LGBM Model, Exchange Rates)
    ES: Inicializa el cliente y las dependencias (Instrucciones, API Key, Modelo LGBM, Tasas de Cambio)
    """
    global API_KEY, genai_client_instance, SYSTEM_INSTRUCTION_FINASSISTENTE
    global lgbm_model, lgbm_label_encoder, lgbm_features, exchange_rates

    # GET CREDENTIALS
    if genai_client_instance is None:
        try:
            with open(CREDENTIALS_PATH, 'r') as f: credentials = json.load(f)
            API_KEY = credentials.get("API_KEY")
            if not API_KEY: raise ValueError(f"{CREDENTIALS_PATH} no contiene 'API_KEY'.")
            genai_client_instance = genai.Client(api_key=API_KEY)
            print("INFO: `google.genai.Client` configurado.")
        # except if credentials paths are not found use secrets of streamlit
        except FileNotFoundError:
            API_KEY = st.secrets["API_KEY"]
            genai_client_instance = genai.Client(api_key=API_KEY)
            print("INFO: `google.genai.Client` configurado con secrets de streamlit.")
        except Exception as e: print(f"ERROR CRÍTICO inicializando `google.genai.Client`: {e}"); raise

    # GET GEMINI API SYSTEM INSTRUCTION
    if not SYSTEM_INSTRUCTION_FINASSISTENTE:
        try:
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f: SYSTEM_INSTRUCTION_FINASSISTENTE = f.read()
            if not SYSTEM_INSTRUCTION_FINASSISTENTE.strip(): raise ValueError(f"{SYSTEM_PROMPT_PATH} está vacío.")
            print(f"INFO: Instrucción del sistema cargada.")
        except Exception as e: print(f"ERROR CRÍTICO cargando instrucción del sistema: {e}"); raise

    # GET EXCHANGE RATES
    if exchange_rates['USD'] == 4200 : # Charge the exchange rates if they are default or not loaded
        try:
            import yfinance as yf
            ticker_usd_cop = yf.Ticker('USDCOP=X'); hist_usd = ticker_usd_cop.history(period='1d')
            if not hist_usd.empty: exchange_rates['USD'] = hist_usd['Close'].iloc[-1]
            ticker_eur_cop = yf.Ticker('EURCOP=X'); hist_eur = ticker_eur_cop.history(period='1d')
            if not hist_eur.empty: exchange_rates['EUR'] = hist_eur['Close'].iloc[-1]
            print(f"INFO: Tasas de cambio actualizadas: USD={exchange_rates['USD']:.2f}, EUR={exchange_rates['EUR']:.2f}")
        except Exception as e: print(f"ADVERTENCIA yfinance: {e}. Usando tasas default.")

    # GET LGBM MODEL (scikit-learn and lightgbm libraries need to be installed)
    if lgbm_model is None:
        try:
            lgbm_bundle = joblib.load(LGBM_MODEL_PATH)
            lgbm_model, lgbm_label_encoder, lgbm_features = lgbm_bundle['model'], lgbm_bundle['label_encoder'], lgbm_bundle['features']
            print(f"INFO: Modelo LGBM cargado. Features esperadas: {lgbm_features}")
        except Exception as e: print(f"ERROR CRÍTICO cargando modelo LGBM: {e}"); raise

# Initialize the client and dependencies
_initialize_client_and_dependencies()
#endregion

# region # --- 4. Chatbot Functions ---
# ES: Funciones para el chatbot
def create_new_chat(model_name: str = "gemini-2.0-flash") -> Optional[Any]:
    """
    EN: Creates a new chat session configured to return JSON according to the Pydantic ChatbotResponse schema.
    
    Args:
        model_name (str, optional): The name of the model to use. Defaults to "gemini-2.0-flash".
    Returns:
        Optional[Any]: The created chat session or None if an error occurred.
    Raises:
        ValueError: If the model name is not valid.
        Exception: If an error occurs while creating the chat session.
    -----
    ES: Crea una nueva sesión de chat configurada para devolver JSON según el schema Pydantic ChatbotResponse.
    El `model_name` debe ser el que te funcionó en tus pruebas (ej: "gemini-2.0-flash").
    
    Args:
        model_name (str, optional): El nombre del modelo a usar. Por defecto es "gemini-2.0-flash".
    Returns:
        Optional[Any]: La sesión de chat creada o None si ocurre un error.
    Raises:
        ValueError: Si el nombre del modelo no es válido.
        Exception: Si ocurre un error al crear la sesión de chat.
    """
    global genai_client_instance
    if not genai_client_instance: print("ERROR: Cliente `google.genai.Client` no inicializado."); return None
    if not SYSTEM_INSTRUCTION_FINASSISTENTE: print("ERROR CRÍTICO: Instrucción del sistema no cargada."); return None

    try:

        model_to_use = model_name
        chat_session = genai_client_instance.chats.create(
            model=model_to_use,
            config=google_genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_FINASSISTENTE,
                response_mime_type="application/json",
                response_schema=ChatbotResponse, # Here is used the Pydantic models
            )
        )
        print(f"INFO: Nueva sesión de chat creada y configurada para JSON con modelo {model_to_use}.")
        return chat_session
    except Exception as e:
        print(f"ERROR al crear nueva sesión de chat configurada: {e}")
        import traceback; traceback.print_exc(); return None

def send_message_to_chat(
    chat_session: Any, # Objeto devuelto por client.chats.create()
    user_message: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]: # Devuelve un diccionario, no un objeto Pydantic
    """
    EN: Sends a message to the chat session and returns the response as a dictionary.
    
    Args:
        chat_session (Any): The chat session object returned by `client.chats.create()`.
        user_message (str): The message to send to the chat session.
    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: A tuple containing the response as a dictionary and an error message if any.
    Raises:
        ValueError: If the chat session is not valid.
        Exception: If an error occurs while sending the message.
    ----- 
    ES: Envía un mensaje a la sesión de chat y devuelve la respuesta como un diccionario.
    
    Args:
        chat_session (Any): El objeto de la sesión de chat devuelto por `client.chats.create()`. 
        user_message (str): El mensaje a enviar a la sesión de chat.
    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: Una tupla que contiene la respuesta como un diccionario y un mensaje de error si lo hay.
    Raises:
        ValueError: Si la sesión de chat no es válida.
        Exception: Si ocurre un error al enviar el mensaje.
    """
    if not chat_session:
        return None, "Error: La sesión de chat no es válida (None)."

    raw_response_text_for_logging = "No se obtuvo respuesta de texto de la API."
    try:
        # print(f"DEBUG: Enviando a `chat.send_message`: '{user_message}'")
        api_response = chat_session.send_message(user_message)

        if not hasattr(api_response, 'text') or not api_response.text:
            error_msg = "Respuesta de la API no contiene 'text' o está vacío."
            # print(f"ERROR: {error_msg}")
            return None, error_msg

        raw_response_text_for_logging = api_response.text
        cleaned_json_text = raw_response_text_for_logging.strip()
        # print(f"DEBUG: Raw JSON text from model: {cleaned_json_text}")

        # Remove code block markers if present
        if cleaned_json_text.startswith("```json"):
            cleaned_json_text = cleaned_json_text[len("```json"):]
        if cleaned_json_text.endswith("```"):
            cleaned_json_text = cleaned_json_text[:-len("```")]
        cleaned_json_text = cleaned_json_text.strip()

        if not cleaned_json_text:
            error_msg = "El contenido JSON de la respuesta está vacío después de limpiar."
            # print(f"ERROR: {error_msg}")
            return None, error_msg
            
        # Try to parse the cleaned JSON text with json.loads
        parsed_dict_response = json.loads(cleaned_json_text)
        
        # Minimum validation of the dictionary structure (optional, but recommended)
        if not isinstance(parsed_dict_response, dict) or \
           "interaction_type" not in parsed_dict_response or \
           "message_for_user" not in parsed_dict_response or \
           "conversation_state" not in parsed_dict_response:
            error_msg = f"El JSON recibido no tiene la estructura esperada. Recibido: {parsed_dict_response}"
            # print(f"ERROR: {error_msg}")
            return None, error_msg
            
        return parsed_dict_response, None # Return the parsed dictionary

    except json.JSONDecodeError as json_err:
        error_msg = f"Error al decodificar JSON: {json_err}. Respuesta cruda: {raw_response_text_for_logging}"
        # print(f"ERROR: {error_msg}")
        return None, error_msg
    except Exception as e_api: 
        error_msg = f"Error inesperado durante send_message: {e_api}"
        # print(f"ERROR: {error_msg}")
        import traceback; traceback.print_exc()
        return None, error_msg


def _convert_to_cop(value_str: str, original_currency: Optional[str]) -> float:
    """
    EN: Converts a value from a given currency to COP (Colombian Peso).
    
    Args:
        value_str (str): The value to convert as a string.
        original_currency (Optional[str]): The original currency code (e.g., "USD", "EUR").
    Returns:
        float: The converted value in COP.
    Raises:
        ValueError: If the value_str is not a valid number.
        TypeError: If the value_str is not a string.
    ----- 
    ES: Convierte un valor de una moneda dada a COP (Peso Colombiano).
    
    Args:
        value_str (str): El valor a convertir como una cadena.
        original_currency (Optional[str]): El código de la moneda original (por ejemplo, "USD", "EUR").
    Returns:
        float: El valor convertido en COP.
    Raises:
        ValueError: Si el value_str no es un número válido.
        TypeError: Si el value_str no es una cadena.
    """
    try: value = float(value_str)
    except (ValueError, TypeError):
        print(f"ADVERTENCIA: Valor '{value_str}' no es numérico válido para conversión. Devolviendo 0.")
        return 0.0
    currency_code = str(original_currency).upper().strip() if original_currency else "COP"
    if not currency_code: currency_code = "COP"
    if currency_code == "COP": return value
    rate = exchange_rates.get(currency_code)
    if rate: return value * rate
    print(f"ADVERTENCIA: No se encontró tasa para {original_currency}. Devolviendo valor original.")
    return value

def run_lgbm_analysis(conversation_state_dict: Dict[str, Any]) -> Optional[str]:
    """
    EN: Prepares data from the `conversation_state` dictionary and runs the LightGBM model.
    Args:
        conversation_state_dict (Dict[str, Any]): The conversation state dictionary.
    Returns:
        Optional[str]: The analysis result as a string or None if an error occurs.
    Raises:
        ValueError: If the conversation_state_dict is not valid.
        Exception: If an error occurs while running the LightGBM model.
    ----- 
    ES: Prepara datos desde el diccionario `conversation_state` y ejecuta el modelo LightGBM.
    Args:
        conversation_state_dict (Dict[str, Any]): El diccionario de estado de la conversación.
    Returns:
        Optional[str]: El resultado del análisis como una cadena o None si ocurre un error.
    Raises:
        ValueError: Si el conversation_state_dict no es válido.
        Exception: Si ocurre un error al ejecutar el modelo LightGBM.
    """
    global lgbm_model, lgbm_label_encoder, lgbm_features, genai_client_instance

    if not (lgbm_model and lgbm_label_encoder and lgbm_features):
        return "Error: Modelo de Análisis no disponible"
    if not genai_client_instance:
        return "Error: Cliente `google.genai.Client` no disponible para inferir sector."

    # Access to the fields from the conversation_state_dict
    completed_fields_list = conversation_state_dict.get("completed_fields", [])
    if not isinstance(completed_fields_list, list): # Validation just in case
        print(f"ERROR: 'completed_fields' no es una lista en conversation_state: {completed_fields_list}")
        return "Error: Formato de datos internos incorrecto para análisis."

    # Convert List[Dict] (now it's JSON directly) to a map for easy access
    completed_data_map: Dict[str, Dict[str, Any]] = {
        cf_dict["name"]: cf_dict for cf_dict in completed_fields_list if isinstance(cf_dict, dict) and "name" in cf_dict
    }
    # print(f"\n--- Formateando datos para LightGBM --- Mapa de campos completados: {completed_data_map}")

    def get_field_attr(field_name: str, attribute: str, default_val: Any = None) -> Any:
        field_dict = completed_data_map.get(field_name)
        return field_dict.get(attribute, default_val) if field_dict else default_val

    def get_field_currency(field_name: str) -> str:
        field_dict = completed_data_map.get(field_name)
        return field_dict.get("currency", "COP") if field_dict else "COP"

    ing_act_val_str = get_field_attr("ingresos_o_activos", "value", "0")
    ing_act_mon = get_field_currency("ingresos_o_activos")
    cart_val_str = get_field_attr("valor_cartera", "value", "0")
    cart_mon = get_field_currency("valor_cartera")
    deud_val_str = get_field_attr("valor_deudas", "value", "0")
    deud_mon = get_field_currency("valor_deudas")
    area_empresa = get_field_attr("area_categoria", "value", "Desconocida")
    num_empleados_str = get_field_attr('numero_empleados', "value", "0")
    
    try: num_empleados_float = float(num_empleados_str)
    except ValueError:
        print(f"ADVERTENCIA: `numero_empleados` ('{num_empleados_str}') no es un número. Usando 0.")
        num_empleados_float = 0.0

    sector = get_field_attr("sector", "value", "Desconocido")
    # print(f"INFO: Sector obtenido del estado: {sector}")
    if sector == "Desconocido" and area_empresa != "Desconocida":
        # print(f"ADVERTENCIA: Sector no provisto o 'Desconocido'. Infiriendo de '{area_empresa}'.")
        try:
            system_instruction_sector = "solo devuelve el sector al que pertenece el area, valores posibles: Primario, Secundario, Terciario, Cuaternario. Si no es claro, responde 'Desconocido'."
            contents_for_sector = f"Área de la empresa: \"{area_empresa}\""
            response_sector_gen = genai_client_instance.models.generate_content(
                model="gemini-2.0-flash", contents=contents_for_sector,
                config=google_genai_types.GenerateContentConfig(
                    system_instruction=system_instruction_sector, response_mime_type="text/plain"
                ))
            sector_text = response_sector_gen.text.strip()
            if sector_text in ["Primario", "Secundario", "Terciario", "Cuaternario", "Desconocido"]: sector = sector_text
            else:
                for s_val in ["Primario", "Secundario", "Terciario", "Cuaternario"]:
                    if s_val.lower() in sector_text.lower(): sector = s_val; break
            # print(f"INFO: Sector (fallback one-shot) para '{area_empresa}': {sector}")
        except Exception as e_s: print(f"ERROR extrayendo sector (fallback): {e_s}")

    data_para_df = {
        'Sector': [sector], 'Area': [area_empresa],
        'Numero Empleados': [num_empleados_float],
        'Activos (COP)': [_convert_to_cop(ing_act_val_str, ing_act_mon)],
        'Cartera (COP)': [_convert_to_cop(cart_val_str, cart_mon)],
        'Deudas (COP)': [_convert_to_cop(deud_val_str, deud_mon)],
    }
    df_row = pd.DataFrame.from_dict(data_para_df)

    for col in lgbm_features:
        if col not in df_row.columns:
            df_row[col] = 'Desconocido' if df_row[col].dtype.name in ['object', 'category'] else 0.0
        if col in ['Numero Empleados', 'Activos (COP)', 'Cartera (COP)', 'Deudas (COP)']:
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(0.0)
        if df_row[col].dtype.name == 'object':
            df_row[col] = df_row[col].astype('category')
        elif not pd.api.types.is_numeric_dtype(df_row[col]):
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(0.0)

    df_row = df_row[lgbm_features]
    # print(f"DataFrame final para LGBM:\n{df_row.head()}\nTipos:\n{df_row.dtypes}")

    prediction_encoded = lgbm_model.predict(df_row)[0]
    prediction_label = lgbm_label_encoder.inverse_transform([prediction_encoded])[0]
    return str(prediction_label)


def generate_final_message_with_advice(lgbm_category: str, company_name: Optional[str]) -> str:
    """
    EN: Generates a final message with advice based on the LGBM category.
    Args:
        lgbm_category (str): The category predicted by the LGBM model.
        company_name (Optional[str]): The name of the company, if available.
    Returns:
        str: The final message with advice.
    Raises:
        ValueError: If the lgbm_category is not one of the expected values.
        Exception: If an unexpected error occurs.
    -----
    ES: Genera un mensaje final con consejos basados en la categoría del LGBM.
    Args:
        lgbm_category (str): La categoría predicha por el modelo LGBM.
        company_name (Optional[str]): El nombre de la empresa, si está disponible.
    Returns:
        str: El mensaje final con consejos.
    Raises:
        ValueError: Si la categoría del LGBM no es una de las esperadas.
        Exception: Si ocurre un error inesperado.
    """
    
    company_str = f"para **{company_name}**" if company_name else "para tu empresa"
    base_message = f"### 📊 Resultado del Análisis Financiero Preliminar {company_str}\n\n"
    base_message += f"Tras analizar los datos, la clasificación preliminar de la salud financiera es: **{lgbm_category}**.\n\n"
    advice = ""
    if lgbm_category == "En Quiebra Técnica / Insolvente":
        advice = """
        **Acciones Urgentes Recomendadas:** 😟
        * 🚨 **Busca Asesoría Profesional Inmediata:** Contacta a un asesor financiero especializado en reestructuraciones o insolvencia y a un abogado mercantil.
        * 🔍 **Análisis Profundo de Viabilidad:** Evalúa si existe alguna posibilidad de reestructurar la deuda y las operaciones.
        * 🛑 **Control Estricto de Gastos:** Minimiza todos los gastos no esenciales.
        * 🤝 **Comunicación con Acreedores:** Considera iniciar conversaciones transparentes con tus principales acreedores.
        *Esta es una situación muy delicada que requiere acción inmediata y experta.*
        """
    elif lgbm_category == "Crítica / Muy Débil":
        advice = """
        **Pasos Clave para Fortalecer:** 📉
        * 💼 **Revisión Urgente de Deudas:** Analiza la estructura de tu deuda (costos, plazos) y explora opciones de refinanciación o consolidación.
        * 💰 **Mejora del Flujo de Caja:** Implementa medidas para acelerar cobros y optimizar pagos.
        * ✂️ **Reducción de Costos:** Identifica áreas donde se puedan reducir costos operativos sin afectar la actividad principal.
        * 💡 **Plan de Viabilidad:** Desarrolla un plan financiero detallado para revertir la situación. Considera buscar capital adicional si es viable.
        *Es crucial actuar con prontitud para evitar un deterioro mayor.*
        """
    elif lgbm_category == "Vulnerable / Débil":
        advice = """
        **Sugerencias para Mejorar la Resiliencia:** 🤔
        * 📊 **Optimización de la Deuda:** Evalúa si tu nivel de endeudamiento es el óptimo. Busca mejorar las condiciones de tus créditos actuales.
        * 📈 **Incremento de Rentabilidad:** Analiza tus márgenes de ganancia y busca formas de mejorarlos.
        * 🛡️ **Creación de un Fondo de Reserva:** Comienza a construir o aumenta un colchón financiero para imprevistos.
        * 🧐 **Monitoreo Continuo:** Sigue de cerca tus indicadores financieros clave (liquidez, endeudamiento, rentabilidad).
        *Pequeños ajustes proactivos pueden marcar una gran diferencia.*
        """
    elif lgbm_category == "Estable / Regular":
        advice = """
        **Recomendaciones para Mantener y Crecer:** 👍
        * ✅ **Sigue las Buenas Prácticas:** Continúa con una gestión financiera prudente.
        * 🌱 **Explora Oportunidades de Crecimiento Controlado:** Considera inversiones o expansiones que no comprometan tu estabilidad actual.
        * 🔄 **Optimización Continua:** Siempre hay espacio para mejorar la eficiencia operativa y financiera.
        * 📊 **Planificación a Largo Plazo:** Asegúrate de tener planes financieros que miren hacia el futuro.
        *¡Vas por buen camino! La clave es la consistencia y la adaptación.*
        """
    elif lgbm_category == "Sólida / Buena":
        advice = """
        **Estrategias para Potenciar tu Fortaleza:** 🎉
        * 🚀 **Inversión Estratégica:** Con una base financiera sólida, puedes considerar inversiones estratégicas para acelerar el crecimiento, innovar o diversificar.
        * 💪 **Fortalecimiento de la Ventaja Competitiva:** Utiliza tu solidez para mejorar tu posición en el mercado.
        * 🌟 **Atracción de Talento e Inversión:** Una buena salud financiera es atractiva para empleados clave e inversores.
        * 🛡️ **Gestión de Riesgos Proactiva:** Aunque estés bien, no descuides la planificación ante posibles cambios en el entorno.
        *¡Excelente gestión! Aprovecha esta posición para asegurar un futuro aún más brillante.*
        """
    elif lgbm_category == "Excelente / Muy Sólida":
        advice = """
        **Maximizando una Posición de Élite:** 🏆
        * 🌍 **Liderazgo e Innovación:** Tu posición te permite liderar en innovación y explorar nuevas fronteras.
        * 🤝 **Oportunidades de Adquisición o Expansión Mayor:** Considera movimientos estratégicos que capitalicen tu fortaleza financiera.
        * 📈 **Optimización del Retorno sobre el Capital:** Asegúrate de que tus activos y capital estén generando el mejor retorno posible.
        * 🌟 **Legado y Sostenibilidad:** Piensa en cómo mantener esta excelencia a muy largo plazo y el impacto que puedes generar.
        *¡Felicidades por una gestión financiera impecable! Eres un referente.*
        """
    else: # Unknown category
        advice = "No tengo consejos específicos para esta categoría, pero te recomiendo revisar tus finanzas detalladamente con un profesional. 👍"
    final_message = base_message + advice + f"\n\n¡Mucha suerte con {company_name}! Espero que este análisis preliminar te sea de utilidad. 😊"
    
    # Remove extra quotes and spaces
    final_message = final_message.replace('"""', "")
    final_message = re.sub(r'[ \t]+', ' ', final_message)
    # print(f"\n--- Mensaje Final para el Usuario ---\n{final_message}")
    return final_message
#endregion