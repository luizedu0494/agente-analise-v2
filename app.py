# app.py - Versão Final com Agente ReAct

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
import re
from PIL import Image

# Importações para LangChain e Groq
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent, Tool

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq e Streamlit")

# --- Ferramenta de Plotagem (Backend) ---
def python_plot_tool(code_to_exec: str) -> str:
    df = st.session_state.get('df_global')
    if df is None:
        return "Erro: Dataframe não encontrado."
    
    plt.close('all')
    try:
        local_namespace = {
            "df": df, "plt": plt, "sns": __import__("seaborn"), "pd": pd
        }
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

# --- Lógica da Interface (Frontend) ---

if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "df_global" not in st.session_state:
    st.session_state.df_global = None

with st.sidebar:
    st.header("1. Carregue seu arquivo")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if st.button("Resetar Sessão"):
        st.session_state.clear()
        st.rerun()

    if uploaded_file is not None and st.session_state.agent is None:
        with st.spinner("Carregando arquivo e inicializando agente..."):
            try:
                st.session_state.df_global = pd.read_csv(uploaded_file)
                groq_api_key = st.secrets["GROQ_API_KEY"]
                
                llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                tools = [
                    Tool(
                        name="PythonPlotter",
                        func=python_plot_tool,
                        description=(
                            "Use esta ferramenta para gerar gráficos e visualizações de dados em Python. "
                            "Um dataframe pandas já está carregado na variável 'df'. "
                            "Você deve escrever o código Python para gerar UM ÚNICO gráfico. "
                            "Não tente gerar múltiplos gráficos em um loop. "
                            "O código será executado e o caminho do arquivo do gráfico será retornado."
                        )
                    )
                ]
                
                # --- AQUI ESTÁ A CORREÇÃO FINAL ---
                # Trocamos o agente para um tipo universalmente compatível.
                st.session_state.agent = initialize_agent(
                    tools, 
                    llm, 
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # Usando o agente ReAct
                    verbose=True, 
                    handle_parsing_errors=True
                )
                st.success("Agente pronto para uso!")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

st.header("2. Converse com seus dados")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

if prompt := st.chat_input("Faça sua pergunta sobre o arquivo..."):
    if st.session_state.agent is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral primeiro.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando e respondendo..."):
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    output_text = response.get("output", "A resposta do agente foi vazia.")
                    
                    image_path = None
                    match = re.search(r"(/tmp/[a-zA-Z0-9/_-]+\.(png|jpg|jpeg))", output_text)
                    if match:
                        image_path = match.group(1)

                    st.markdown(output_text)
                    st.session_state.history.append({"role": "assistant", "content": output_text})

                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption="Gráfico gerado pelo agente")
                        st.session_state.history.append({"role": "assistant", "image": image})
                    elif image_path:
                        st.error(f"O agente mencionou um gráfico em '{image_path}', mas o arquivo não foi encontrado.")

                except Exception as e:
                    error_message = f"Ocorreu um erro durante a execução do agente: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
