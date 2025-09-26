import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
import matplotlib.pyplot as plt
import io

# Configuração da página do Streamlit
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (E.D.A.)")

# Inicializa o estado da sessão
if 'history' not in st.session_state:
    st.session_state.history = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'plot_buffer' not in st.session_state:
    st.session_state.plot_buffer = None

# --- FERRAMENTA PERSONALIZADA PARA PLOTAGEM ---
def python_plot_tool(code_to_exec):
    """
    Uma ferramenta que executa código Python para gerar um gráfico com matplotlib
    e o salva em um buffer de memória para ser exibido no Streamlit.
    """
    try:
        # Cria um namespace local para a execução do código
        local_namespace = {"plt": plt, "df": st.session_state.df}
        # Executa o código fornecido pelo agente
        exec(code_to_exec, local_namespace)
        
        # Salva a figura em um buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.session_state.plot_buffer = buf # Armazena o buffer na sessão
        plt.clf() # Limpa a figura para a próxima plotagem
        return "Gráfico gerado com sucesso e está pronto para ser exibido."
    except Exception as e:
        return f"Erro ao executar o código de plotagem: {e}"

# Função para criar e configurar o agente
def criar_agente(df, api_key):
    """Cria um agente LangChain para interagir com um DataFrame Pandas."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        # CORREÇÃO: Embrulha a função de plotagem em um objeto Tool
        plot_tool = Tool(
            name="PlottingTool",
            func=python_plot_tool,
            description="""
            Use esta ferramenta para gerar gráficos e visualizações de dados.
            O input deve ser um código Python que usa a variável 'df' (o DataFrame)
            e a biblioteca matplotlib (importada como 'plt').
            O código não deve chamar 'plt.show()'. O gráfico será salvo automaticamente.
            Exemplo de input: 'plt.hist(df["nome_da_coluna"])'
            """
        )

        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            extra_tools=[plot_tool] # Passa a ferramenta corretamente formatada
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
        st.subheader("Faça uma pergunta sobre seus dados:")
        query = st.text_input("Ex: 'Gere um histograma para a coluna Amount'", key="query_input")

        if query:
            with st.spinner("O agente está pensando..."):
                st.session_state.plot_buffer = None # Limpa o buffer antigo
                response = agente.invoke(query)
                output_text = response.get("output", "Sem resposta.")
                st.session_state.history.append({"pergunta": query, "resposta": output_text})
                st.rerun() # Recarrega a página para exibir o histórico atualizado

# --- Exibição do Histórico ---
st.subheader("Histórico da Conversa")
if not st.session_state.history:
    st.info("Nenhuma pergunta foi feita ainda.")
else:
    # Itera na ordem normal para exibir o mais recente por último
    for i, chat in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(chat["pergunta"])
        with st.chat_message("assistant"):
            st.write(chat["resposta"])
            # Se um gráfico foi gerado na última interação, exibe-o
            if i == len(st.session_state.history) - 1 and st.session_state.plot_buffer:
                st.image(st.session_state.plot_buffer, caption="Gráfico Gerado pelo Agente")
                st.download_button(
                    label="Baixar Gráfico",
                    data=st.session_state.plot_buffer,
                    file_name=f"grafico_{i}.png",
                    mime="image/png"
                )

if uploaded_file is None:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
