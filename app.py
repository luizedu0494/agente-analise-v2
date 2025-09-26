import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Importações para a solução definitiva
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool
import google.generativeai as genai

# Configuração da página
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (E.D.A.)")

# Inicialização do estado da sessão
if 'history' not in st.session_state:
    st.session_state.history = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'plot_buffer' not in st.session_state:
    st.session_state.plot_buffer = None

# Ferramenta de plotagem
def python_plot_tool(code_to_exec: str) -> str:
    if st.session_state.df is None:
        return "Erro: Nenhum DataFrame carregado."
    try:
        local_namespace = {"df": st.session_state.df, "plt": plt}
        exec(code_to_exec, local_namespace)
        fig = plt.gcf()
        if fig.get_axes():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.session_state.plot_buffer = buf
            plt.clf()
            return "Gráfico gerado com sucesso."
        else:
            return "Código executado, mas nenhum gráfico foi gerado."
    except Exception as e:
        plt.clf()
        return f"Erro ao executar código de plotagem: {e}"

# Função para criar o agente
def criar_agente(df, api_key):
    try:
        # --- SOLUÇÃO DEFINITIVA ---
        # 1. Configurar a API Key diretamente na biblioteca do Google.
        genai.configure(api_key=api_key)

        # 2. Criar o LLM usando o modelo estável 'gemini-pro'.
        #    Não passamos mais a chave aqui, pois já foi configurada.
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            convert_system_message_to_human=True
        )
        
        plot_tool = Tool(
            name="PlottingTool",
            func=python_plot_tool,
            description="Use esta ferramenta para gerar gráficos. O input deve ser código Python usando 'df' e 'plt'. Não chame 'plt.show()'."
        )

        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            extra_tools=[plot_tool]
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o agente: {e}")
        return None

# --- Interface do Usuário ---
st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

if st.sidebar.button("Limpar Sessão"):
    st.session_state.clear()
    st.rerun()

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
    
    df = st.session_state.df
    st.subheader("Visualização dos Dados")
    st.dataframe(df.head())

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Chave de API do Google não encontrada nos segredos.")
        st.stop()

    agente = criar_agente(df, api_key)

    # Exibe o histórico
    for chat in st.session_state.get('history', []):
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
            if "plot" in chat:
                st.image(chat["plot"], caption="Gráfico Gerado")

    if agente:
        if query := st.chat_input("Faça uma pergunta sobre seus dados..."):
            st.session_state.history.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("O agente está pensando..."):
                    st.session_state.plot_buffer = None
                    try:
                        response = agente.invoke(query)
                        output_text = response.get("output", "Sem resposta.")
                        
                        assistant_response = {"role": "assistant", "content": output_text}
                        if st.session_state.plot_buffer:
                            assistant_response["plot"] = st.session_state.plot_buffer
                        
                        st.session_state.history.append(assistant_response)
                        st.rerun()

                    except Exception as e:
                        error_message = f"Ocorreu um erro: {e}"
                        st.error(error_message)
                        st.session_state.history.append({"role": "assistant", "content": error_message})

else:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
