# app.py - Versão Final com Ferramenta Modificada (e o 'df' corrigido)

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
                # Passamos o dataframe para a ferramenta, para que ela possa executá-lo
                python_tool = PythonAstREPLTool(locals={"df": df})

                # 2. Criamos nossa função "wrapper"
                def run_python_repl_wrapper(query: str) -> str:
                    """
                    Executa o código Python e intercepta saídas de gráficos,
                    forçando uma resposta útil em português.
                    """
                    if "plt.show()" in query or "savefig" in query:
                        try:
                            # Executa o código usando a ferramenta original
                            # A ferramenta já tem acesso ao 'df'
                            result = python_tool.run(query)
                            
                            # Mesmo que o resultado seja o objeto matplotlib,
                            # o importante é que o arquivo foi salvo.
                            # Vamos procurar o arquivo de plot.
                            plot_dir = "/tmp/streamlit_plots"
                            if not os.path.exists(plot_dir) or not os.listdir(plot_dir):
                                return "Tentei gerar um gráfico, mas não encontrei o diretório de plots ou ele está vazio."

                            # Pega o arquivo mais recente no diretório
                            files = sorted(
                                [os.path.join(plot_dir, f) for f in os.listdir(plot_dir)],
                                key=os.path.getmtime
                            )
                            new_plot_path = files[-1]
                            return f"Gráfico gerado com sucesso e salvo em: {new_plot_path}"
                        except Exception as e:
                            return f"Erro ao tentar gerar o gráfico: {e}"
                    else:
                        # Se não for um gráfico, executa normalmente
                        return python_tool.run(query)

                # 3. Criamos uma nova ferramenta que usa nossa função wrapper
                custom_python_tool = Tool(
                    name="python_repl_ast", # O nome deve ser o mesmo da ferramenta interna
                    func=run_python_repl_wrapper,
                    description="Uma ferramenta para executar código python para análise de dados com pandas."
                )

                # 4. Criamos o agente, passando a NOSSA ferramenta customizada
                # --- INÍCIO DA CORREÇÃO ---
                st.session_state.agent_executor = create_pandas_dataframe_agent(
                    llm=st.session_state.llm,
                    df=df, # O ARGUMENTO QUE FALTAVA!
                    tool=[custom_python_tool], 
                    verbose=True,
                    agent_type="openai-tools" # Especificar o tipo de agente
                )
                # --- FIM DA CORREÇÃO ---
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
