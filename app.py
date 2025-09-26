import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuração da página do Streamlit
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (E.D.A.)")

# Função para criar e configurar o agente
def criar_agente(df, api_key):
    """Cria um agente LangChain para interagir com um DataFrame Pandas."""
    try:
        # Inicializa o modelo de linguagem (LLM) do Google Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",
            google_api_key=api_key,
            temperature=0, # Usamos 0 para respostas mais diretas e menos "criativas"
            # --- A LINHA OBSOLETA FOI REMOVIDA DAQUI ---
        )

        # Cria o agente de DataFrame. É aqui que a "mágica" acontece.
        # Ele conecta o LLM (Gemini) ao nosso DataFrame (df)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True, # Mostra os "pensamentos" do agente no terminal
            allow_dangerous_code=True # Permite que o agente execute código Python gerado por ele
        )
        return agent
    except Exception as e:
        st.error(f"Erro ao inicializar o agente: {e}")
        return None

# --- Interface do Usuário ---

st.sidebar.header("Configurações")
# Uploader de arquivo na barra lateral
uploaded_file = st.sidebar.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Carrega o arquivo CSV em um DataFrame do Pandas
        df = pd.read_csv(uploaded_file)
        st.subheader("Visualização dos Dados Carregados")
        st.dataframe(df.head()) # Mostra as 5 primeiras linhas do dataframe

        # Pega a chave da API dos segredos do Streamlit
        api_key = st.secrets["GOOGLE_API_KEY"]

        # Cria o agente com os dados carregados
        agente = criar_agente(df, api_key)

        if agente:
            st.subheader("Faça uma pergunta sobre seus dados:")
            # Caixa de texto para o usuário fazer a pergunta
            query = st.text_input("Ex: 'Qual a distribuição da coluna Class?' ou 'Gere um histograma para a coluna Amount'")

            if query:
                with st.spinner("O agente está pensando e processando sua solicitação..."):
                    try:
                        # Executa o agente com a pergunta do usuário
                        # O método .invoke() envia a pergunta para o agente processar
                        response = agente.invoke(query)

                        st.subheader("Resposta do Agente:")
                        # A resposta do agente pode conter texto e gráficos.
                        # O Streamlit é inteligente para renderizar diferentes tipos de saída.
                        st.write(response)

                    except Exception as e:
                        st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")

    except Exception as e:
        st.error(f"Erro ao carregar ou processar o arquivo CSV: {e}")
else:
    st.info("Por favor, faça o upload de um arquivo CSV para começar.")