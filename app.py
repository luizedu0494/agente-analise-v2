# app.py - Versão Final com Memória de Conversa

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
st.title("🤖 Agente de Análise de Dados Conversacional")

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
                st.success("Pronto! Inicie sua conversa.")
            except Exception as e:
                st.error(f"Erro na inicialização: {e}")

# --- Área de Chat ---
st.header("2. Converse com seus dados")

# Exibe o histórico de mensagens
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
    if st.session_state.df is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral primeiro.")
    else:
        # Adiciona e exibe a pergunta do usuário
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Gera e exibe a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Pensando e gerando código..."):
                
                # --- INÍCIO DA LÓGICA DE MEMÓRIA ---

                # 1. Formata o histórico de conversa para incluir no prompt
                formatted_history = ""
                for message in st.session_state.history:
                    role = "Usuário" if message["role"] == "user" else "Assistente (código gerado)"
                    formatted_history += f"{role}: {message['content']}\n"

                # 2. Cria o novo prompt que inclui o histórico
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
                
                # --- FIM DA LÓGICA DE MEMÓRIA ---

                code_response = st.session_state.llm.invoke(code_generation_prompt)
                generated_code = code_response.content.strip().replace("```python", "").replace("```", "").strip()
                
                if "plt.show()" not in generated_code and "print(" not in generated_code:
                    generated_code = f"print({generated_code})"

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
                        
                        # Adiciona o código gerado ao histórico, não a imagem
                        st.session_state.history.append({"role": "assistant", "content": generated_code})

                    else:
                        with contextlib.redirect_stdout(output_buffer):
                            exec(generated_code, {"df": df})
                        
                        text_output = output_buffer.getvalue().strip()
                        
                        # Adiciona o código gerado ao histórico
                        st.session_state.history.append({"role": "assistant", "content": generated_code})
                        
                        # Exibe o resultado do código
                        final_response_text = f"**Resultado:**\n```\n{text_output}\n```"
                        st.markdown(final_response_text)

                except Exception as e:
                    error_message = f"Ocorreu um erro ao executar o código gerado: {e}"
                    st.error(error_message)
                    # Adiciona a mensagem de erro ao histórico para contexto futuro
                    st.session_state.history.append({"role": "assistant", "content": f"Erro: {error_message}"})
