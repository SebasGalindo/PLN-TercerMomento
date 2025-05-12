import streamlit as st

pages = {
    "Principal": [
        st.Page("homepage.py", title="Taller Tercer Momento AG"),
        st.Page("chatbot_streamlit.py", title="Chatbot"),
    ],
    "Explicaciones": [
        st.Page("explanation_train_lightgbm.py", title="Entrenamiento para LightGBM"),
        st.Page("explanation_dataset_lightgbm.py", title="Dataset para LightGBM"),
        st.Page("explanation_chatbot.py", title="Explicación del Chatbot"),
    ],
}

pg = st.navigation(pages)
pg.run()

