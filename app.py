# app.py - Versão Final Definitiva (Estilo Gradio no Streamlit)

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
from langchain.agents import Tool

# --- Ferramenta de Plotagem Customizada (Nosso Controle Total) ---
def python_plot_tool(code_to_exec: str) -> str:
    """
    Executa código Python para gerar UM ÚNICO gráfico.
    Salva o gráfico em um arquivo temporário e retorna o caminho do arquivo.
    """
    df = st.session_state.get('df_global')
    if df is None:
        return "Erro: Dataframe não encontrado."
    
    plt.close('all')
    try:
        # Prepara o ambiente de execução com todas as bibliotecas necessárias
        local_namespace = {
            "df": df, "plt": plt, "sns": __import__("seaborn"), "pd": pd
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
            # Retorna uma mensagem clara com o caminho do arquivo
            return f"Plot gerado com sucesso e salvo em: {file_path}"
        
        plt.close(fig)
        return "Nenhum plot foi gerado pelo código."
    except Exception as e:
        plt.close('all')
        print(f"Erro ao executar o código de plotagem: {e}")
        return f"Erro ao executar o código: {e}"

# --- Lógica da Interface (Frontend) ---

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq")

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
                df = pd.read_csv(uploaded_file)
                st.session_state.df_global = df
                
                groq_api_key = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                # Criamos a lista de ferramentas, incluindo a nossa ferramenta customizada
                tools = [
                    Tool(
                        name="PythonPlotter",
                        func=python_plot_tool,
                        description=(
                            "Use esta ferramenta para gerar um gráfico ou visualização de dados. "
                            "O input deve ser um código Python válido. "
                            "Um dataframe pandas já está disponível como 'df'. "
                            "Seu código NÃO deve tentar gerar múltiplos gráficos em um loop. "
                            "Foque em gerar UMA ÚNICA figura por vez."
                        )
                    )
                ]
                
                # Criamos o agente e passamos nossa ferramenta em `extra_tools`
                st.session_state.agent_executor = create_pandas_dataframe_agent(
                    llm,
                    df,
                    agent_type="openai-tools",
                    extra_tools=tools, # Forçando o agente a conhecer nossa ferramenta
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                )
                st.success("Agente pronto! Faça sua pergunta.")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

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
                    # Usamos a regex para encontrar o caminho do nosso plot customizado
                    match = re.search(r"(/tmp/streamlit_plots/.*\.png)", output_text)
                    if match:
                        image_path = match.group(1)

                    st.markdown(output_text)
                    st.session_state.history.append({"role": "assistant", "content": output_text})

                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption="Gráfico gerado pelo agente")
                    elif "Plot gerado com sucesso" in output_text and not image_path:
                         st.warning("O agente parece ter gerado um gráfico, mas não consegui encontrar o caminho do arquivo na resposta.")

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
