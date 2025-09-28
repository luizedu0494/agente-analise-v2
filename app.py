# app.py - Versão Final com Widgets Interativos e Memória

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib

from langchain_groq import ChatGroq

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados Interativo")

# --- Estado da Sessão ---
if "history" not in st.session_state:
    st.session_state.history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# --- Barra Lateral ---
with st.sidebar:
    st.header("1. Carregue seu arquivo")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if st.button("Resetar Sessão"):
        st.session_state.clear()
        st.rerun()

    if uploaded_file is not None and st.session_state.df is None:
        with st.spinner("Carregando arquivo e inicializando..."):
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                groq_api_key = st.secrets["GROQ_API_KEY"]
                st.session_state.llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
                st.success("Pronto! Use os widgets ou inicie uma conversa.")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

# --- Interface Principal ---

# Só exibe as ferramentas de análise se um dataframe estiver carregado
if st.session_state.df is not None:
    df = st.session_state.df
    column_options = df.columns.tolist()

    # --- INÍCIO DA NOVA SEÇÃO DE WIDGETS ---
    st.header("1. Análise Rápida com Widgets")
    
    # Usando colunas para organizar os widgets
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de uma Coluna")
        hist_col = st.selectbox("Escolha uma coluna para o histograma:", column_options, key="hist_col")
        hist_bins = st.slider("Número de bins:", min_value=5, max_value=200, value=30, key="hist_bins")
        
        if st.button("Gerar Histograma"):
            fig, ax = plt.subplots()
            ax.hist(df[hist_col], bins=hist_bins)
            ax.set_title(f"Distribuição de {hist_col}")
            ax.set_xlabel(hist_col)
            ax.set_ylabel("Frequência")
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("Relação entre Duas Colunas")
        scatter_x = st.selectbox("Coluna para o Eixo X:", column_options, key="scatter_x")
        scatter_y = st.selectbox("Coluna para o Eixo Y:", column_options, index=1 if len(column_options) > 1 else 0, key="scatter_y")
        
        if st.button("Gerar Gráfico de Dispersão"):
            fig, ax = plt.subplots()
            ax.scatter(df[scatter_x], df[scatter_y])
            ax.set_title(f"Relação entre {scatter_x} e {scatter_y}")
            ax.set_xlabel(scatter_x)
            ax.set_ylabel(scatter_y)
            st.pyplot(fig)
            plt.close(fig)
            
    st.markdown("---") # Divisor visual
    # --- FIM DA NOVA SEÇÃO DE WIDGETS ---


    # --- Seção de Chat Conversacional (continua a mesma) ---
    st.header("2. Converse com o Assistente (para análises complexas)")

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando e gerando código..."):
                formatted_history = ""
                for message in st.session_state.history:
                    role = "Usuário" if message["role"] == "user" else "Assistente (código gerado)"
                    formatted_history += f"{role}: {message['content']}\n"

                code_generation_prompt = f"""
                Você é um especialista em Python e pandas. Continue a conversa abaixo gerando o próximo bloco de código Python necessário para responder à última pergunta do usuário.
                Considere todo o histórico da conversa para entender o contexto.

                ### Histórico da Conversa ###
                {formatted_history}
                ### Fim do Histórico ###

                Baseado na última pergunta do usuário e no contexto acima, gere o próximo código Python.
                - Para perguntas que retornam um valor (cálculos, dtypes, describe), **SEMPRE** use a função `print()`.
                - Para perguntas que pedem um gráfico, use `plt.show()`.
                - Forneça apenas o bloco de código Python, sem explicações.
                """
                
                code_response = st.session_state.llm.invoke(code_generation_prompt)
                generated_code = code_response.content.strip().replace("```python", "").replace("```", "").strip()
                
                if "plt.show()" not in generated_code and "print(" not in generated_code:
                    generated_code = f"print({generated_code})"

                st.write("Código a ser executado:")
                st.code(generated_code)

            with st.spinner("Executando código e preparando resposta..."):
                output_buffer = io.StringIO()
                try:
                    if "plt.show()" in generated_code:
                        fig, ax = plt.subplots()
                        exec(generated_code.replace("plt.show()", ""), {"df": df, "plt": plt, "ax": ax})
                        st.pyplot(fig)
                        plt.close(fig)
                        st.session_state.history.append({"role": "assistant", "content": generated_code})
                    else:
                        with contextlib.redirect_stdout(output_buffer):
                            exec(generated_code, {"df": df})
                        text_output = output_buffer.getvalue().strip()
                        st.session_state.history.append({"role": "assistant", "content": generated_code})
                        final_response_text = f"**Resultado:**\n```\n{text_output}\n```"
                        st.markdown(final_response_text)
                except Exception as e:
                    error_message = f"Ocorreu um erro ao executar o código gerado: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": f"Erro: {error_message}"})
else:
    st.info("👆 Carregue um arquivo CSV na barra lateral para começar a análise.")

