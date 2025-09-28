# app.py - Versão Final com Tradução Inteligente

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

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
if "llm" not in st.session_state:
    st.session_state.llm = None

# --- Função Auxiliar para Decidir se Traduz ---
def is_natural_language(text):
    """Verifica se o texto parece ser linguagem natural e não código ou objeto."""
    # Se for muito curto, provavelmente não é uma frase para traduzir
    if len(text.split()) < 3:
        return False
    # Se contém '<' e '>' ou '/', provavelmente é um objeto ou caminho
    if '<' in text and '>' in text or '/' in text:
        return False
    # Se contém palavras comuns de respostas em inglês, é um bom candidato
    common_words = ['mean', 'median', 'column', 'data', 'following', 'there is', 'are']
    if any(word in text.lower() for word in common_words):
        return True
    # Se nada disso se aplicar, mas for longo o suficiente, vamos tentar traduzir
    if len(text.split()) >= 3:
        return True
    return False

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
                st.session_state.llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                st.session_state.agent_executor = create_pandas_dataframe_agent(
                    llm=st.session_state.llm,
                    df=df,
                    agent_type="openai-tools", 
                    verbose=True,
                    allow_dangerous_code=True,
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
                    original_output = response.get("output", "A resposta do agente foi vazia.")
                    
                    final_output = original_output

                    # --- CAMADA DE TRADUÇÃO INTELIGENTE ---
                    # 1. Decidimos se devemos ou não traduzir
                    if is_natural_language(original_output):
                        with st.spinner("Traduzindo resposta para português..."):
                            translation_prompt = f"Traduza o seguinte texto para o português do Brasil, mantendo a formatação e o significado originais:\n\n{original_output}"
                            translation_response = st.session_state.llm.invoke(translation_prompt)
                            final_output = translation_response.content
                    # --- FIM DA CAMADA DE TRADUÇÃO ---

                    # A busca pelo gráfico continua na saída original, que é mais confiável
                    image_path = None
                    match = re.search(r"(/tmp/plots/.*\.png)", original_output)
                    if not match:
                        # Alguns agentes podem retornar um texto diferente, vamos tentar outra regex
                        match = re.search(r"Plot saved to (.*\.png)", original_output)

                    if match:
                        image_path = match.group(1)
                        # Se o agente retornou o caminho, vamos criar uma mensagem mais amigável
                        if final_output == original_output: # Só substitui se não foi traduzido
                            final_output = f"Aqui está o gráfico que você pediu. Ele foi gerado e salvo em `{image_path}`."

                    st.markdown(final_output)
                    st.session_state.history.append({"role": "assistant", "content": final_output})

                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption="Gráfico gerado pelo agente")
                    elif "plot" in original_output.lower() and not image_path:
                         st.warning("O agente mencionou um gráfico, mas não consegui encontrá-lo ou exibi-lo.")

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
