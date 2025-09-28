# app.py - Versão Final Corrigida (Sem extra_tools)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

# Importações para LangChain e Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq")

# --- Estado da Sessão ---
if "history" not in st.session_state:
    st.session_state.history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "df_global" not in st.session_state:
    st.session_state.df_global = None

# --- Barra Lateral ---
with st.sidebar:
    st.header("1. Carregue seu arquivo")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if st.button("Resetar Sessão"):
        st.session_state.clear()
        st.rerun()

    if uploaded_file is not None and st.session_state.agent_executor is None:
        with st.spinner("Carregando arquivo e inicializando agente..."):
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_global = df
                
                groq_api_key = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                # Criamos o agente sem o argumento extra_tools
                st.session_state.agent_executor = create_pandas_dataframe_agent(
                    llm,
                    df, # Passa o dataframe diretamente
                    agent_type="openai-tools", 
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                )
                st.success("Agente pronto! Faça sua pergunta.")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

# --- Área de Chat ---
st.header("2. Converse com seus dados")
st.info("Para melhores resultados, peça um tipo de gráfico por vez (ex: 'gere um histograma para V1').")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça uma pergunta específica..."):
    if st.session_state.agent_executor is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral primeiro.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando e respondendo..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": prompt})
                    output_text = response.get("output", "A resposta do agente foi vazia.")
                    
                    image_path = None
                    # A regex para encontrar o caminho do plot do agente
                    match = re.search(r"(/tmp/plots/.*\.png)", output_text)
                    if match:
                        image_path = match.group(1)

                    st.markdown(output_text)
                    st.session_state.history.append({"role": "assistant", "content": output_text})

                    if image_path and os.path.exists(image_path):
                        image = Image.Image.open(image_path)
                        st.image(image, caption="Gráfico gerado pelo agente")
                    elif "plot has been saved" in output_text and not image_path:
                         st.warning("O agente gerou um gráfico, mas não consegui exibi-lo. Tente pedir o gráfico novamente de forma mais específica.")

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
