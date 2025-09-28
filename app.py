# app.py - Versão Final com Camada de Tradução Explícita

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

# Importações para LangChain e Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# Não precisamos mais de prompts customizados, vamos simplificar

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq")

# --- Estado da Sessão ---
if "history" not in st.session_state:
    st.session_state.history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "df_global" not in st.session_state:
    st.session_state.df_global = None
# Adicionamos o LLM ao estado da sessão para reutilizá-lo na tradução
if "llm" not in st.session_state:
    st.session_state.llm = None

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
                # Guardamos a instância do LLM para usar depois
                st.session_state.llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                # --- VOLTANDO AO SIMPLES ---
                # Criamos o agente da forma mais básica possível, sem tentar forçar o prompt.
                # Deixamos ele responder em inglês.
                st.session_state.agent_executor = create_pandas_dataframe_agent(
                    llm=st.session_state.llm,
                    df=df,
                    agent_type="openai-tools", 
                    verbose=True,
                    allow_dangerous_code=True,
                )
                # --- FIM DA SIMPLIFICAÇÃO ---

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
                    # 1. O agente executa e responde (provavelmente em inglês)
                    response = st.session_state.agent_executor.invoke({"input": prompt})
                    english_output = response.get("output", "A resposta do agente foi vazia.")
                    
                    # --- CAMADA DE TRADUÇÃO FORÇADA ---
                    # 2. Verificamos se a resposta não está vazia para traduzir
                    if english_output and english_output != "A resposta do agente foi vazia.":
                        with st.spinner("Traduzindo resposta para português..."):
                            # Criamos um prompt de tradução simples e direto
                            translation_prompt = f"Traduza o seguinte texto para o português do Brasil, mantendo a formatação original (como listas e quebras de linha):\n\n{english_output}"
                            
                            # 3. Usamos o mesmo LLM para fazer a tradução
                            translation_response = st.session_state.llm.invoke(translation_prompt)
                            
                            # A resposta final é o conteúdo da tradução
                            final_output = translation_response.content
                    else:
                        final_output = english_output
                    # --- FIM DA CAMADA DE TRADUÇÃO ---

                    # A regex para encontrar o caminho do plot continua funcionando
                    image_path = None
                    # Buscamos o plot tanto na resposta original quanto na traduzida
                    match = re.search(r"(/tmp/plots/.*\.png)", english_output)
                    if match:
                        image_path = match.group(1)

                    # Exibimos a resposta final traduzida
                    st.markdown(final_output)
                    # E salvamos a resposta traduzida no histórico
                    st.session_state.history.append({"role": "assistant", "content": final_output})

                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption="Gráfico gerado pelo agente")
                    elif "plot has been saved" in english_output and not image_path:
                         st.warning("O agente gerou um gráfico, mas não consegui exibi-lo. Tente pedir o gráfico novamente de forma mais específica.")

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
