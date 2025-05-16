# chatbot_streamlit.py
# Este archivo contiene el frontend del chatbot.
# Authors: John Sebastián Galindo Hernández y Miguel Ángel Moreno Beltrán

import streamlit as st
import matplotlib.pyplot as plt

# Try to import the backend and handle errors gracefully

# Add the parent directory to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Path agregado:", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Path actual:", os.getcwd())
print("sys.path:", sys.path)

try:
    import Backend.chatbot_logic as bot
except ModuleNotFoundError as e:
    st.error(f"Error: No se encontró el archivo 'chatbot_logic.py'. {e}")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error inesperado al cargar el módulo del bot: {e}")
    import traceback
    st.error(f"Traceback:\n{traceback.format_exc()}")
    st.stop()

st.set_page_config(page_title="Chatbot Financiero UdeC", layout="centered")
st.title("🤖 Chatbot Financiero UdeC")
st.caption("Análisis preliminar de salud financiera empresarial con IA")


def initialize_new_chat_session():
    """
    EN: Initializes or restarts the chat session and the state in Streamlit.
    ES: Inicializa o reinicia la sesión de chat y el estado en Streamlit.
    """
    
    # Create a new chat session using the `create_new_chat` function from the backend with a flexible model with fast response
    st.session_state.gemini_chat_session = bot.create_new_chat(
        model_name="gemini-2.0-flash" 
    )
    
    # If the chat session could not be created, stop the app
    if not st.session_state.gemini_chat_session:
        st.error("Error crítico: No se pudo crear una nueva sesión de chat con el servicio de IA. Revisa la configuración y la API Key.")
        st.stop() # Detener la app si no se puede crear el chat

    st.session_state.chatbot_state = None # This will be updated with the first response from the bot
    st.session_state.chat_history = []
    st.session_state.analysis_done = False # Flag to control if the final analysis has been shown
    
    # A fixed initial greeting and the bot will respond to the first user input.
    initial_greeting = """
    ¡Hola! 👋 Soy un chatbot diseñado para ayudarte a realizar una **evaluación preliminar** de la salud financiera de tu empresa 📊. 
    Para ello, te haré algunas preguntas clave de forma secuencial. Una vez tenga toda la información, 
    utilizaré un modelo para darte una **clasificación general** que va desde *'Insolvente'* hasta *'Excelente'*. 
    \n\nLas **preguntas** que te haré son:
    \n1.  **nombre** de la empresa.
    \n2.  **Área o categoría** principal de actividad (ej: *'Agricultura'*, *'Tecnología'*, *'Comercio al por menor'*).
    \n3.  **Número total** de empleados.
    \n4.  Valor aproximado de los **ingresos anuales** O el **valor total de los activos** (puedes darme cualquiera de los dos, el que te sea más fácil).
    \n5.  Valor aproximado de la **cartera** o cuentas por cobrar a clientes.
    \n6.  Valor aproximado de las **deudas totales** (bancarias, proveedores, etc.).
    \n\n**¿Cómo responder?** 🤔
    \nTrata de ser lo más claro y directo posible. 
    \n* ✅ **Buen ejemplo (Ingresos):** *\"Tuvimos ingresos por unos 500 millones de pesos el año pasado\"* o *\"Aproximadamente 150 mil dólares\"*.
    \n* ✅ **Buen ejemplo (Categoría):** *\"Somos del sector construcción\"* o *\"Nos dedicamos al desarrollo de software\"*.
    \n* ❌ **Mal ejemplo (Ingresos):** *\"Pues más o menos bien, creo\"* (No da un valor)
    \n* ❌ **Mal ejemplo (Categoría):** *\"Hacemos varias cosas\"* (Muy vago)
    \n\n¡Empecemos cuando quieras! 👍"
    """
    st.session_state.chat_history.append({"role": "assistant", "content": initial_greeting})

# --- Sidebar ---
st.sidebar.title("Opciones")
if st.sidebar.button("✨ Nuevo Chat", key="new_chat_button"):
    initialize_new_chat_session()
    st.rerun() # Force a rerun to refresh the UI with the new state

# --- Initialize Session State ---
if 'gemini_chat_session' not in st.session_state:
    initialize_new_chat_session()

# --- Show Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
        if "bar_chart" in message:
            st.pyplot(message["bar_chart"])

# --- Get User Input ---
# Disable input if final analysis has been shown
user_input_disabled = st.session_state.get('analysis_done', False)

user_prompt = st.chat_input(
    "Escribe tu respuesta aquí...",
    disabled=user_input_disabled,
    key="user_chat_input"
)

if user_input_disabled and not user_prompt: # Si ya terminó y no hay nuevo input
    st.info("Análisis completado. Para un nuevo análisis, presiona '✨ Nuevo Chat' en la barra lateral.")


if user_prompt:
    # Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process user input using the bot logic
    with st.chat_message("assistant"):
        with st.spinner("FinAsistente está pensando... 🧠"):
            # Send message to Gemini
            response_dict, error_msg = bot.send_message_to_chat(
                st.session_state.gemini_chat_session,
                user_prompt
            )

        if error_msg or not response_dict:
            error_display_message = f"Lo siento, ocurrió un error al procesar tu mensaje: {error_msg or 'Respuesta inesperada.'}"
            st.error(error_display_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_display_message})
        else:
            # Show bot response (message_for_user)
            bot_message_content = response_dict.get("message_for_user", "No pude generar una respuesta adecuada.")
            st.markdown(bot_message_content)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_message_content})

            # Update conversation state in session_state
            st.session_state.chatbot_state = response_dict.get("conversation_state")

            # Verify if the conversation has ended and it's time for the LGBM analysis
            interaction_type = response_dict.get("interaction_type")
            if interaction_type == "datos_completos":
                st.session_state.analysis_done = True # Mark that the collection process has ended
                with st.spinner("Todos los datos recolectados. Realizando análisis financiero final... 📊"):
                    # Ensure that chatbot_state (which is the conversation_state) is not None
                    if st.session_state.chatbot_state:
                        lgbm_category = bot.run_lgbm_analysis(st.session_state.chatbot_state)

                        if lgbm_category and "Error" not in lgbm_category:
                            company_name = "tu empresa" # Default
                            # Extraer nombre de la empresa de los completed_fields para personalizar
                            if isinstance(st.session_state.chatbot_state, dict):
                                completed_fields_list = st.session_state.chatbot_state.get("completed_fields", [])
                                for cf in completed_fields_list:
                                    if cf.get("name") == "nombre_empresa":
                                        company_name = cf.get("value", "tu empresa")
                                        break
                            
                            final_analysis_message, final_info = bot.generate_final_message_with_advice(
                                st.session_state.chatbot_state,
                                lgbm_category
                            )
                        else:
                            final_analysis_message = f"Lo siento, hubo un problema al generar el análisis final: {lgbm_category or 'Resultado vacío.'}"
                            final_info = None
                        st.markdown(final_analysis_message) # Show the final analysis result
                        if final_info:
                            print("Mostrando gráfico final...")
                            annual_profit = final_info.get("annual_profit", 0)
                            annual_debts = final_info.get("annual_debts", 0)
                            annual_cartera = final_info.get("annual_cartera", 0)
                            # Generate bar plot
                            fig, ax = plt.subplots(figsize=(6,4))
                            bars = ax.bar(['Utilidad/Ingresos', 'Deudas', 'Cartera'], [annual_profit, annual_debts, annual_cartera], color=['#4CAF50', '#F44336', '#2196F3'])
                            ax.set_ylabel('Valor (COP)')
                            ax.set_title('Resumen Financiero Anual')
                            ax.bar_label(bars, fmt='${:,.0f}')
                            plt.tight_layout()
                            
                        st.session_state.chat_history.append({"role": "assistant", "content": final_analysis_message, "bar_chart": fig})
                        st.rerun() # Forzar rerun para deshabilitar input y mostrar el mensaje final correctamente.
                    else:
                        err_msg_state = "Error: El estado de la conversación no está disponible para el análisis final."
                        st.error(err_msg_state)
                        st.session_state.chat_history.append({"role": "assistant", "content": err_msg_state})

# Add a footer or additional information
st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado como proyecto para PLN. Recuerda que este es un análisis preliminar y no reemplaza la asesoría financiera profesional.")