# app.py - Versão Final com Ferramenta Modificada

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.tools import Tool

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq")

# --- Estado da Sessão ---
# ... (sem alterações no estado da sessão)
if "history" not in st.session_state:
    st.session_state.history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "df_global" not in st.session_state:
    st.session_state.df_global = None
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
                st.session_state.llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                
                # --- INÍCIO DA MODIFICAÇÃO DA FERRAMENTA ---

                # 1. Criamos a ferramenta padrão que o agente usaria
                python_tool = PythonAstREPLTool(
                    locals={"df": df},
                    description="Uma ferramenta para executar código python."
                )

                # 2. Criamos nossa função "wrapper"
                def run_python_repl_wrapper(query: str) -> str:
                    """
                    Executa o código Python e intercepta saídas de gráficos,
                    forçando uma resposta útil em português.
                    """
                    # Verifica se o código é para gerar um gráfico
                    if "plt.show()" in query or "savefig" in query:
                        # Executa o código, mas se prepara para ignorar a saída
                        try:
                            # Onde os gráficos do Streamlit são salvos por padrão
                            plot_dir = "/tmp/streamlit_plots"
                            os.makedirs(plot_dir, exist_ok=True)
                            # Conta quantos arquivos existem para prever o nome do novo
                            num_existing_plots = len(os.listdir(plot_dir))
                            
                            # Executa o código original
                            python_tool.run(query)

                            # Verifica se um novo arquivo foi criado
                            files = sorted(os.listdir(plot_dir), key=lambda x: os.path.getmtime(os.path.join(plot_dir, x)))
                            if len(files) > num_existing_plots:
                                new_plot_path = os.path.join(plot_dir, files[-1])
                                # Retorna a nossa mensagem personalizada em português!
                                return f"Gráfico gerado com sucesso e salvo em: {new_plot_path}"
                            else:
                                return "Tentei gerar um gráfico, mas não consegui confirmar se foi salvo."
                        except Exception as e:
                            return f"Erro ao tentar gerar o gráfico: {e}"
                    else:
                        # Se não for um gráfico, executa normalmente
                        return python_tool.run(query)

                # 3. Criamos uma nova ferramenta que usa nossa função wrapper
                custom_python_tool = Tool(
                    name=python_tool.name,
                    func=run_python_repl_wrapper,
                    description=python_tool.description,
                    args_schema=python_tool.args_schema
                )

                # 4. Criamos o agente, passando a NOSSA ferramenta customizada
                st.session_state.agent_executor = create_pandas_dataframe_agent(
                    llm=st.session_state.llm,
                    # O agente agora usará nossa ferramenta modificada
                    tool=[custom_python_tool], 
                    verbose=True,
                )
                # --- FIM DA MODIFICAÇÃO DA FERRAMENTA ---

                st.success("Agente pronto! Faça sua pergunta.")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

# --- Área de Chat (com tradução inteligente para texto) ---
st.header("2. Converse com seus dados")
st.info("Para melhores resultados, peça um tipo de gráfico por vez (ex: 'gere um histograma para V1').")

def is_text_to_translate(text):
    """Função simplificada para decidir se traduz."""
    text = text.strip()
    if not text or text.startswith("Gráfico gerado com sucesso"):
        return False
    # Se contém palavras comuns de respostas em inglês, é um bom candidato
    common_words = ['mean', 'median', 'column', 'data', 'following', 'there is', 'are', 'is', 'the']
    if any(word in text.lower().split() for word in common_words):
        return True
    return False

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
                    # Traduz apenas se for uma resposta textual em inglês
                    if is_text_to_translate(original_output):
                        with st.spinner("Traduzindo resposta..."):
                            translation_prompt = f"Traduza o seguinte texto para o português do Brasil, mantendo a formatação e o significado originais:\n\n{original_output}"
                            translation_response = st.session_state.llm.invoke(translation_prompt)
                            final_output = translation_response.content
                    
                    image_path = None
                    match = re.search(r"(/tmp/streamlit_plots/.*\.png)", final_output)
                    if match:
                        image_path = match.group(1)

                    st.markdown(final_output)
                    st.session_state.history.append({"role": "assistant", "content": final_output})

                    if image_path and os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption="Gráfico gerado pelo agente")
                    elif "gráfico" in final_output.lower() and not image_path:
                         st.warning("O agente mencionou um gráfico, mas não consegui encontrá-lo ou exibi-lo.")

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
