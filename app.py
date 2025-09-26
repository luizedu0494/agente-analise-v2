# ==============================================================================
# SOLUÇÃO PARA ERROS DE PERMISSÃO EM AMBIENTES DE HOSPEDAGEM
# Estas linhas devem ser as primeiras do arquivo para garantir que sejam executadas antes de tudo.
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
# ==============================================================================

# Importações principais do projeto
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import Tool
import google.generativeai as genai

# --- O CORPO PRINCIPAL DO APLICATIVO ---

# Configuração inicial da página do Streamlit
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (E.D.A.)")

# Inicialização do estado da sessão para persistir dados entre as interações
if 'history' not in st.session_state:
    st.session_state.history = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'plot_buffer' not in st.session_state:
    st.session_state.plot_buffer = None

# Ferramenta de plotagem que o agente usará
def python_plot_tool(code_to_exec: str) -> str:
    """
    Executa código Python para gerar um gráfico com matplotlib e o salva em um buffer de memória.
    O input deve ser um código Python que usa a variável 'df' (o DataFrame)
    e a biblioteca matplotlib (importada como 'plt'). Não chame 'plt.show()'.
    """
    if st.session_state.df is None:
        return "Erro: Nenhum DataFrame carregado. Peça ao usuário para fazer o upload de um arquivo primeiro."
    
    try:
        # Prepara um ambiente seguro para a execução do código gerado pelo agente
        local_namespace = {"df": st.session_state.df, "plt": plt}
        exec(code_to_exec, local_namespace)
        
        # Captura a figura gerada pelo matplotlib
        fig = plt.gcf()
        if fig.get_axes(): # Verifica se há algo desenhado na figura
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.session_state.plot_buffer = buf # Armazena o gráfico na sessão para uso posterior
            plt.clf() # Limpa a figura para a próxima plotagem
            return "Gráfico gerado com sucesso e está pronto para ser exibido."
        else:
            return "O código foi executado, mas nenhum gráfico foi gerado."
    except Exception as e:
        plt.clf() # Garante que a figura seja limpa mesmo em caso de erro
        return f"Erro ao executar o código de plotagem: {e}"

# Função para criar e configurar o agente LangChain
def criar_agente(df, api_key):
    try:
        # Configura a API do Google de forma explícita para evitar erros de versão
        genai.configure(api_key=api_key)
        
        # Inicializa o modelo de linguagem (LLM)
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            convert_system_message_to_human=True
        )
        
        # Cria a ferramenta de plotagem que o agente pode escolher usar
        plot_tool = Tool(
            name="PlottingTool",
            func=python_plot_tool,
            description="Use esta ferramenta para gerar gráficos e visualizações. O input deve ser código Python usando 'df' e 'plt'. Não chame 'plt.show()'."
        )
        
        # Cria o agente que conecta o LLM, o DataFrame e as ferramentas
        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True, # Mostra os "pensamentos" do agente nos logs do servidor
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}, # Ajuda a lidar com respostas mal formatadas do LLM
            extra_tools=[plot_tool]
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o agente: {e}")
        return None

# --- Interface do Usuário com Streamlit ---

st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

if st.sidebar.button("Limpar Sessão"):
    st.session_state.clear()
    st.rerun()

if uploaded_file is not None:
    # Carrega o DataFrame apenas uma vez por sessão para economizar tempo
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
    
    df = st.session_state.df
    st.subheader("Visualização dos Dados")
    st.dataframe(df.head())

    # Pega a chave da API dos segredos da plataforma de hospedagem
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Chave de API do Google (GOOGLE_API_KEY) não encontrada nos segredos.")
        st.stop()

    # Cria o agente com os dados carregados
    agente = criar_agente(df, api_key)

    # Exibe o histórico da conversa
    for chat in st.session_state.get('history', []):
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
            if "plot" in chat:
                st.image(chat["plot"], caption="Gráfico Gerado")

    # Caixa de input para o usuário
    if agente:
        if query := st.chat_input("Faça uma pergunta sobre seus dados..."):
            # Adiciona a pergunta do usuário ao histórico e exibe na tela
            st.session_state.history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            
            # Processa a pergunta com o agente e exibe a resposta
            with st.chat_message("assistant"):
                with st.spinner("O agente está pensando..."):
                    st.session_state.plot_buffer = None # Limpa o buffer de gráfico antigo
                    try:
                        response = agente.invoke(query)
                        output_text = response.get("output", "Não foi possível obter uma resposta.")
                        
                        # Prepara e armazena a resposta completa do assistente
                        assistant_response = {"role": "assistant", "content": output_text}
                        if st.session_state.plot_buffer:
                            assistant_response["plot"] = st.session_state.plot_buffer
                        
                        st.session_state.history.append(assistant_response)
                        st.rerun() # Recarrega a página para exibir a resposta completa de forma limpa

                    except Exception as e:
                        error_message = f"Ocorreu um erro durante a execução: {e}"
                        st.error(error_message)
                        st.session_state.history.append({"role": "assistant", "content": error_message})
else:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")
