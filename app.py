# app.py - Versão para Streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
from PIL import Image

# Importações para LangChain e Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool

# --- Funções de Lógica (Exatamente as mesmas de antes) ---

def python_plot_tool(code_to_exec: str) -> str:
    # Esta função não precisa de nenhuma mudança!
    df = st.session_state.df_global
    if df is None:
        return "Erro: Dataframe não encontrado."
    plt.close('all')
    try:
        if code_to_exec.strip().startswith("```"):
            code_to_exec = code_to_exec.strip()[3:-3].strip()
            if code_to_exec.lower().startswith('python'):
                code_to_exec = code_to_exec[6:].strip()
        local_namespace = {"df": df, "plt": plt}
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

# --- Lógica da Interface com Streamlit ---

# Configuração da página
st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq e Streamlit")

# Gerenciamento de estado da sessão do Streamlit
if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "df_global" not in st.session_state:
    st.session_state.df_global = None

# Barra Lateral para Upload e Configuração
with st.sidebar:
    st.header("1. Carregue seu arquivo")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None and st.session_state.agent is None:
        with st.spinner("Carregando arquivo e inicializando agente..."):
            try:
                st.session_state.df_global = pd.read_csv(uploaded_file)
                groq_api_key = st.secrets["GROQ_API_KEY"] # Streamlit usa st.secrets
                
                llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                plot_tool = Tool(
                    name="PlottingTool",
                    func=python_plot_tool,
                    description=(
                        "Use esta ferramenta para gerar gráficos. "
                        "IMPORTANTE: Um dataframe pandas já está carregado na variável 'df'. "
                        "NÃO carregue os dados novamente com 'pd.read_csv()'. "
                        "Use diretamente a variável 'df'. "
                        "A ferramenta retornará o caminho do arquivo onde o gráfico foi salvo."
                    )
                )
                
                st.session_state.agent = create_pandas_dataframe_agent(
                    llm,
                    st.session_state.df_global,
                    verbose=True,
                    allow_dangerous_code=True,
                    max_iterations=5,
                    agent_executor_kwargs={"handle_parsing_errors": True},
                    extra_tools=[plot_tool]
                )
                st.success("Agente pronto para uso!")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

# Área Principal para Chat
st.header("2. Converse com seus dados")

# Exibe o histórico do chat
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

# Entrada do usuário
if prompt := st.chat_input("Faça sua pergunta sobre o arquivo..."):
    if st.session_state.agent is None:
        st.warning("Por favor, carregue um arquivo CSV primeiro.")
    else:
        # Adiciona a pergunta do usuário ao histórico e exibe
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Invoca o agente
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    output_text = response.get("output", "Não obtive uma resposta.")
                    
                    # Verifica se a resposta é um gráfico
                    if output_text.strip().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = output_text.strip()
                        if os.path.exists(file_path):
                            image = Image.open(file_path)
                            st.image(image)
                            st.session_state.history.append({"role": "assistant", "image": image})
                        else:
                            st.error(f"Agente retornou um caminho de imagem inválido: {file_path}")
                            st.session_state.history.append({"role": "assistant", "content": f"Erro: Caminho da imagem não encontrado."})
                    else:
                        st.markdown(output_text)
                        st.session_state.history.append({"role": "assistant", "content": output_text})

                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")
                    st.session_state.history.append({"role": "assistant", "content": f"Erro: {e}"})

