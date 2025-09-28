# app.py - Versão Final para Streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
import re  # Importa a biblioteca de expressões regulares
from PIL import Image

# Importações para LangChain e Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool

# --- Funções de Lógica (Backend - Nenhuma mudança aqui) ---

def python_plot_tool(code_to_exec: str) -> str:
    """
    Executa código Python para gerar gráficos.
    Salva o gráfico em um arquivo temporário e retorna o caminho do arquivo.
    """
    # Usa o dataframe do estado da sessão do Streamlit
    df = st.session_state.get('df_global')
    if df is None:
        return "Erro: Dataframe não encontrado."
    
    plt.close('all')
    try:
        if code_to_exec.strip().startswith("```"):
            code_to_exec = code_to_exec.strip()[3:-3].strip()
            if code_to_exec.lower().startswith('python'):
                code_to_exec = code_to_exec[6:].strip()

        local_namespace = {"df": df, "plt": plt, "sns": __import__("seaborn")}
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

# Configuração da página
st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq e Streamlit")

# Gerenciamento de estado da sessão para manter os dados entre as interações
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

    # Botão para reiniciar a sessão e carregar um novo arquivo
    if st.button("Resetar e Carregar Novo Arquivo"):
        st.session_state.history = []
        st.session_state.agent = None
        st.session_state.df_global = None
        st.rerun()

    # Inicializa o agente APENAS se um arquivo for carregado e o agente não existir
    if uploaded_file is not None and st.session_state.agent is None:
        with st.spinner("Carregando arquivo e inicializando agente..."):
            try:
                st.session_state.df_global = pd.read_csv(uploaded_file)
                # Carrega a chave de API dos segredos do Streamlit Cloud
                groq_api_key = st.secrets["GROQ_API_KEY"]
                
                llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                plot_tool = Tool(
                    name="PlottingTool",
                    func=python_plot_tool,
                    description=(
                        "Use esta ferramenta para gerar gráficos e visualizações de dados. "
                        "IMPORTANTE: Um dataframe pandas já está carregado na variável 'df'. "
                        "Você NÃO DEVE carregar os dados novamente com 'pd.read_csv()'. "
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

# Área Principal para o Chat
st.header("2. Converse com seus dados")

# Exibe o histórico da conversa
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

# Captura a entrada do usuário
if prompt := st.chat_input("Faça sua pergunta sobre o arquivo..."):
    if st.session_state.agent is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral primeiro.")
    else:
        # Adiciona e exibe a pergunta do usuário
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processa a resposta do agente
        with st.chat_message("assistant"):
            with st.spinner("Analisando e respondendo..."):
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    output_text = response.get("output", "Não obtive uma resposta de texto.")
                    
                    # --- AQUI ESTÁ A CORREÇÃO PRINCIPAL ---
                    image_path = None
                    
                    # 1. Procura por um caminho de arquivo de imagem na resposta do agente
                    match = re.search(r"(/tmp/[a-zA-Z0-9/_-]+\.(png|jpg|jpeg))", output_text)
                    if match:
                        image_path = match.group(1)

                    # 2. Sempre exibe a resposta de texto do agente
                    st.markdown(output_text)
                    st.session_state.history.append({"role": "assistant", "content": output_text})

                    # 3. Se um caminho de imagem foi encontrado E o arquivo existe, exibe a imagem
                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption="Gráfico gerado pelo agente")
                        # Adiciona a imagem ao histórico para que ela não desapareça
                        st.session_state.history.append({"role": "assistant", "image": image})
                    elif image_path:
                        st.error(f"O agente mencionou um gráfico em '{image_path}', mas o arquivo não foi encontrado.")

                except Exception as e:
                    error_message = f"Ocorreu um erro durante a execução do agente: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
