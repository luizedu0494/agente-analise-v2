# app.py - Versão Final Corrigida para Streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
import re
from PIL import Image

# Importações para LangChain e Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor

# --- Funções de Lógica (Backend) ---

def python_plot_tool(code_to_exec: str) -> str:
    """
    Executa código Python para gerar gráficos.
    Salva o gráfico em um arquivo temporário e retorna o caminho do arquivo.
    """
    df = st.session_state.get('df_global')
    if df is None:
        return "Erro: Dataframe não encontrado."
    
    plt.close('all')
    try:
        # Prepara o ambiente de execução para o código do agente
        local_namespace = {
            "df": df,
            "plt": plt,
            "sns": __import__("seaborn")
        }
        # Executa o código gerado pelo agente
        exec(code_to_exec, local_namespace)
        
        fig = plt.gcf()
        if fig.get_axes():
            temp_dir = "/tmp/streamlit_plots"
            os.makedirs(temp_dir, exist_ok=True)
            file_name = f"plot_{uuid.uuid4()}.png"
            file_path = os.path.join(temp_dir, file_name)
            fig.savefig(file_path)
            plt.close(fig)
            return f"Plot gerado com sucesso e salvo em: {file_path}"
        
        plt.close(fig)
        return "Nenhum plot foi gerado pelo código."
    except Exception as e:
        plt.close('all')
        print(f"Erro ao executar o código de plotagem: {e}")
        return f"Erro ao executar o código: {e}"

# --- Lógica da Interface com Streamlit (Frontend) ---

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq e Streamlit")

# Gerenciamento de estado da sessão
if "history" not in st.session_state:
    st.session_state.history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "df_global" not in st.session_state:
    st.session_state.df_global = None

with st.sidebar:
    st.header("1. Carregue seu arquivo")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if st.button("Resetar Sessão"):
        st.session_state.clear()
        st.rerun()

    if uploaded_file is not None and st.session_state.agent_executor is None:
        with st.spinner("Carregando arquivo e inicializando agente..."):
            try:
                st.session_state.df_global = pd.read_csv(uploaded_file)
                groq_api_key = st.secrets["GROQ_API_KEY"]
                
                llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                # A criação do agente agora é feita aqui, uma única vez.
                st.session_state.agent_executor = create_pandas_dataframe_agent(
                    llm,
                    st.session_state.df_global, # Passa o dataframe aqui
                    verbose=True,
                    allow_dangerous_code=True,
                    agent_executor_kwargs={"handle_parsing_errors": True},
                )
                st.success("Agente pronto para uso!")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

st.header("2. Converse com seus dados")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Faça sua pergunta sobre o arquivo..."):
    if st.session_state.agent_executor is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral primeiro.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando e respondendo..."):
                try:
                    # --- AQUI ESTÁ A MUDANÇA CRUCIAL ---
                    # Em vez de criar uma nova ferramenta, usamos o agente já existente
                    # que "conhece" o dataframe.
                    response = st.session_state.agent_executor.invoke({
                        "input": prompt,
                        # Forçamos o agente a usar o nosso dataframe em cada chamada
                        "df": st.session_state.df_global 
                    })
                    output_text = response.get("output", "Não obtive uma resposta de texto.")
                    
                    st.markdown(output_text)
                    st.session_state.history.append({"role": "assistant", "content": output_text})

                except Exception as e:
                    error_message = f"Ocorreu um erro durante a execução do agente: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})

