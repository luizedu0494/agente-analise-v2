import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
import matplotlib.pyplot as plt
import io

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
    """
    Executa código Python para gerar um gráfico com matplotlib e o salva em um buffer.
    O input deve ser um código Python que usa a variável 'df' (o DataFrame)
    e a biblioteca matplotlib (importada como 'plt').
    """
    if st.session_state.df is None:
        return "Erro: Nenhum DataFrame carregado. Faça o upload de um arquivo CSV."
    
    try:
        local_namespace = {"plt": plt, "df": st.session_state.df, "io": io}
        exec(code_to_exec, local_namespace)
        
        buf = local_namespace.get("buf")
        if isinstance(buf, io.BytesIO):
            st.session_state.plot_buffer = buf
            return "Gráfico gerado com sucesso e está pronto para ser exibido."
        else:
            # Se o código não criou um buffer, tentamos salvar a figura atual
            fig = plt.gcf()
            if fig.get_axes(): # Verifica se há algo na figura
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.session_state.plot_buffer = buf
                plt.clf()
                return "Gráfico gerado com sucesso e está pronto para ser exibido."
            else:
                return "O código foi executado, mas nenhum gráfico foi gerado."

    except Exception as e:
        return f"Erro ao executar o código de plotagem: {e}"

# Função para criar o agente
def criar_agente(df, api_key):
    try:
        # CORREÇÃO FINAL: Especificar a versão da API e o modelo correto.
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", # Usando o modelo mais robusto
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        plot_tool = Tool(
            name="PlottingTool",
            func=python_plot_tool,
            description="""Use esta ferramenta para gerar gráficos e visualizações de dados. O input deve ser um código Python que usa a variável 'df' (o DataFrame) e a biblioteca matplotlib (importada como 'plt'). O código não deve chamar 'plt.show()'. Exemplo de input: 'plt.hist(df["nome_da_coluna"])'"""
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

if st.sidebar.button("Limpar Histórico"):
    st.session_state.history = []
    st.session_state.plot_buffer = None
    st.rerun()

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
    
    df = st.session_state.df
    st.subheader("Visualização dos Dados")
    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Baixar CSV", csv, f"{uploaded_file.name}", "text/csv")

    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Chave de API do Google não encontrada nos segredos.")
        st.stop()

    agente = criar_agente(df, api_key)

    if agente:
        query = st.chat_input("Faça uma pergunta sobre seus dados...")

        if query:
            st.session_state.history.append({"role": "user", "content": query})
            with st.spinner("O agente está pensando..."):
                st.session_state.plot_buffer = None
                response = agente.invoke(query)
                output_text = response.get("output", "Sem resposta.")
                st.session_state.history.append({"role": "assistant", "content": output_text})
                st.rerun()

# Exibição do Histórico
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# Exibe o gráfico se ele foi gerado na última resposta
if st.session_state.plot_buffer:
    with st.chat_message("assistant"):
        st.image(st.session_state.plot_buffer, caption="Gráfico Gerado pelo Agente")
        st.download_button(
            label="Baixar Gráfico",
            data=st.session_state.plot_buffer,
            file_name="grafico_gerado.png",
            mime="image/png"
        )
    st.session_state.plot_buffer = None # Limpa depois de exibir

if uploaded_file is None:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
