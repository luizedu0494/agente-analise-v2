# app.py - Versão Final com 'print()' forçado

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
import io
import contextlib

from langchain_groq import ChatGroq

st.set_page_config(page_title="🤖 Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados com Groq")

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
                st.success("Pronto! Faça sua pergunta.")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

# --- Área de Chat ---
st.header("2. Converse com seus dados")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Faça uma pergunta específica..."):
    if st.session_state.df is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral primeiro.")
    else:
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gerando código Python..."):
                # --- INÍCIO DA CORREÇÃO NO PROMPT ---
                code_generation_prompt = f"""
                Você é um especialista em Python e pandas. Sua tarefa é gerar código para responder a uma pergunta sobre um dataframe chamado 'df'.
                - Para perguntas que retornam um valor, uma série ou um dataframe (como cálculos, descrições ou tipos de dados), **SEMPRE** use a função `print()` para exibir o resultado. Exemplo: `print(df.head())`, `print(df['coluna'].mean())`, `print(df.dtypes)`.
                - Para perguntas que pedem um gráfico, use `plt.show()`.
                - Forneça apenas o bloco de código Python, sem explicações.

                Pergunta: "{user_prompt}"
                """
                # --- FIM DA CORREÇÃO NO PROMPT ---
                code_response = st.session_state.llm.invoke(code_generation_prompt)
                generated_code = code_response.content.strip().replace("```python", "").replace("```", "").strip()
                
                # --- INÍCIO DA MELHORIA DE ROBUSTEZ ---
                # Se o código não for um plot e não tiver print, adiciona um print().
                if "plt.show()" not in generated_code and "print(" not in generated_code:
                    generated_code = f"print({generated_code})"
                # --- FIM DA MELHORIA DE ROBUSTEZ ---

                st.write("Código a ser executado:")
                st.code(generated_code)

            with st.spinner("Executando código e preparando resposta..."):
                df = st.session_state.df
                output_buffer = io.StringIO()
                
                try:
                    if "plt.show()" in generated_code:
                        fig, ax = plt.subplots()
                        exec(generated_code.replace("plt.show()", ""), {"df": df, "plt": plt, "ax": ax})
                        st.pyplot(fig)
                        plt.close(fig)
                        final_response_text = "Aqui está o gráfico que você pediu."
                        st.session_state.history.append({"role": "assistant", "content": final_response_text})

                    else:
                        with contextlib.redirect_stdout(output_buffer):
                            exec(generated_code, {"df": df})
                        
                        text_output = output_buffer.getvalue().strip()
                        
                        # Agora, a saída não estará mais vazia
                        final_response_text = f"**Resultado:**\n```\n{text_output}\n```"

                        st.markdown(final_response_text)
                        st.session_state.history.append({"role": "assistant", "content": final_response_text})

                except Exception as e:
                    error_message = f"Ocorreu um erro ao executar o código gerado: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
