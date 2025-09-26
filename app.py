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
        return "Erro: Nenhum DataFrame carregado."
    
    try:
        # Prepara o ambiente para execução do código do agente
        local_namespace = {"df": st.session_state.df, "plt": plt}
        exec(code_to_exec, local_namespace)
        
        # Tenta salvar a figura gerada
        fig = plt.gcf()
        if fig.get_axes():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.session_state.plot_buffer = buf
            plt.clf() # Limpa a figura para a próxima plotagem
            return "Gráfico gerado com sucesso e está pronto para ser exibido."
        else:
            return "O código foi executado, mas nenhum gráfico foi gerado."
    except Exception as e:
        plt.clf() # Garante que a figura seja limpa mesmo em caso de erro
        return f"Erro ao executar o código de plotagem: {e}"

# Função para criar o agente
def criar_agente(df, api_key):
    try:
        # SOLUÇÃO: Usar o modelo padrão 'gemini-pro' que é estável e compatível.
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        plot_tool = Tool(
            name="PlottingTool",
            func=python_plot_tool,
            description="""Use esta ferramenta APENAS para gerar gráficos e visualizações. O input deve ser um código Python que usa a variável 'df' (o DataFrame) e a biblioteca matplotlib (importada como 'plt'). O código NÃO deve chamar 'plt.show()'. O gráfico será salvo automaticamente."""
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
    st.session_state.df = None # Limpa também o dataframe
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

    if agente:
        query = st.chat_input("Faça uma pergunta sobre seus dados...")

        if query:
            st.session_state.history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("O agente está pensando..."):
                    st.session_state.plot_buffer = None
                    try:
                        response = agente.invoke(query)
                        output_text = response.get("output", "Sem resposta.")
                        st.write(output_text)
                        st.session_state.history.append({"role": "assistant", "content": output_text})

                        # Se um gráfico foi gerado, exibe-o
                        if st.session_state.plot_buffer:
                            st.image(st.session_state.plot_buffer, caption="Gráfico Gerado")
                            st.download_button(
                                "Baixar Gráfico",
                                st.session_state.plot_buffer,
                                "grafico.png",
                                "image/png"
                            )
                    except Exception as e:
                        st.error(f"Ocorreu um erro: {e}")
                        # Verifica se o erro é de cota
                        if "ResourceExhausted" in str(e) or "429" in str(e):
                            st.warning("Você atingiu o limite de requisições da API do Google. Por favor, aguarde um minuto ou verifique seu plano e cobrança na plataforma do Google AI.")
                        st.session_state.history.append({"role": "assistant", "content": f"Erro: {e}"})

else:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")

