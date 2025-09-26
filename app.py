import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib.pyplot as plt
import io

# Configuração da página do Streamlit
st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")
st.title("🤖 Agente Autônomo para Análise de Dados (E.D.A.)")

# Inicializa o estado da sessão para o histórico e o dataframe
if 'history' not in st.session_state:
    st.session_state.history = []
if 'df' not in st.session_state:
    st.session_state.df = None

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
        # O agente agora tem acesso ao pyplot (plt) para criar e salvar gráficos
        return create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}, # Lida melhor com erros de formatação
            extra_tools=[plt.show] # Fornece ao agente a ferramenta de plotagem
        )
    except Exception as e:
        st.error(f"Erro ao inicializar o agente: {e}")
        return None

# --- Interface do Usuário ---

st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

# Botão para limpar o histórico na barra lateral
if st.sidebar.button("Limpar Histórico da Conversa"):
    st.session_state.history = []
    st.rerun()

if uploaded_file is not None:
    # Carrega o dataframe apenas uma vez
    if st.session_state.df is None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo CSV: {e}")
            st.stop()

    df = st.session_state.df
    st.subheader("Visualização dos Dados Carregados")
    st.dataframe(df.head())

    # --- Funcionalidade de Download ---
    st.sidebar.header("Exportar Dados")
    # Converte o dataframe para CSV para o download
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Baixar CSV Original",
        data=csv,
        file_name=f"{uploaded_file.name.split('.')[0]}_original.csv",
        mime="text/csv",
    )


    # Pega a chave da API
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("Chave de API do Google não encontrada. Configure-a nos segredos do Streamlit.")
        st.stop()

    agente = criar_agente(df, api_key)

    if agente:
        st.subheader("Faça uma pergunta sobre seus dados:")
        query = st.text_input("Ex: 'Gere um histograma para a coluna Amount' ou 'Qual a correlação entre as colunas?'", key="query_input")

        if query:
            with st.spinner("O agente está pensando e processando..."):
                try:
                    # Instrução para o agente salvar o gráfico
                    prompt = f"""
                    Analise a seguinte pergunta: '{query}'.
                    Se a pergunta pedir para gerar um gráfico ou visualização, por favor, use a biblioteca matplotlib para criar o gráfico e salve-o em um arquivo chamado 'plot.png'.
                    Responda à pergunta e, se um gráfico for criado, mencione que ele está disponível para download.
                    """
                    response = agente.invoke(prompt)
                    output_text = response.get("output", "Não foi possível obter uma resposta.")

                    # Adiciona a interação ao histórico
                    st.session_state.history.append({"pergunta": query, "resposta": output_text})

                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
                    output_text = None

# --- Exibição do Histórico da Conversa ---
st.subheader("Histórico da Conversa")
if not st.session_state.history:
    st.info("Nenhuma pergunta foi feita ainda.")
else:
    for i, chat in enumerate(reversed(st.session_state.history)):
        with st.chat_message("user"):
            st.write(chat["pergunta"])
        with st.chat_message("assistant"):
            st.write(chat["resposta"])
            # Verifica se um gráfico foi gerado e oferece o download
            try:
                # Salva a figura atual do matplotlib em um buffer de memória
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                
                # Oferece o botão de download para a imagem no buffer
                st.download_button(
                    label="Baixar Gráfico Gerado",
                    data=buf,
                    file_name=f"grafico_{i}.png",
                    mime="image/png"
                )
                plt.clf() # Limpa a figura para a próxima plotagem
            except Exception as e:
                # Se não houver gráfico para salvar, apenas ignora
                pass

if uploaded_file is None:
    st.info("Por favor, faça o upload de um arquivo CSV para começar a análise.")
