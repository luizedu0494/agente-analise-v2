import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Importa a lógica do nosso agente do arquivo separado
from agent_logic import criar_agente_pandas, criar_prompt_detalhado

# --- 1. Configuração da Página e Chaves de API ---
st.set_page_config(
    page_title="Agente de Análise v2.0",
    page_icon="📊",
    layout="wide"
)
load_dotenv()
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# --- 2. Interface do Usuário (UI) ---
st.title("🤖 Agente de Análise de Dados v2.0")
st.markdown("Estrutura renovada para máxima performance e estabilidade.")

uploaded_file = st.file_uploader(
    "**Faça o upload de um arquivo CSV**",
    type=["csv"]
)

# --- 3. Lógica Principal do App ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso! O DataFrame possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
        st.dataframe(df.head()) # Mostra apenas as 5 primeiras linhas para economizar espaço

        st.markdown("---")
        question = st.text_input(
            "**Faça sua pergunta sobre os dados:**",
            placeholder="Ex: Qual a média da coluna 'valor' para cada 'categoria'?"
        )

        if st.button("Analisar Dados", type="primary"):
            if not question:
                st.warning("Por favor, digite uma pergunta para análise.")
            elif not API_KEY:
                st.error("Chave da API do Google não configurada! Verifique seus 'Secrets' no Streamlit Cloud.")
            else:
                with st.spinner("O Agente Gemini está analisando... Isso pode levar um momento. 🧠"):
                    try:
                        # 1. Cria a instância do agente
                        agent_executor = criar_agente_pandas(df, API_KEY)
                        
                        # 2. Cria o prompt detalhado
                        prompt_formatado = criar_prompt_detalhado(question, uploaded_file.name)
                        
                        # 3. Executa o agente com o prompt
                        response = agent_executor.invoke(prompt_formatado)
                        
                        # 4. Exibe a resposta
                        st.success("Análise Concluída!")
                        st.markdown("### Resposta do Agente:")
                        st.write(response["output"])

                    except Exception as e:
                        st.error(f"Ocorreu um erro durante a execução do agente: {e}")

    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
