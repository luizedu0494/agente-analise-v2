# app.py - Abordagem Final e Direta

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
        # Renderiza imagens ou texto
        if "image" in message:
            st.image(message["image"], caption=message.get("content", ""))
        else:
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
                # 1. Gerar o código Python
                code_generation_prompt = f"""
                Você é um especialista em Python, pandas e matplotlib.
                Baseado na pergunta do usuário, escreva um código Python para analisar o dataframe chamado 'df'.
                - Se a pergunta for sobre um cálculo (média, contagem, etc.), use print() para mostrar o resultado.
                - Se a pergunta pedir um gráfico (histograma, dispersão, etc.), use plt.show().
                - Não adicione explicações, apenas o bloco de código Python.

                Pergunta: "{user_prompt}"
                """
                code_response = st.session_state.llm.invoke(code_generation_prompt)
                generated_code = code_response.content.strip().replace("```python", "").replace("```", "").strip()
                
                st.write("Código gerado pelo LLM:")
                st.code(generated_code)

            with st.spinner("Executando código e preparando resposta..."):
                # 2. Executar o código e capturar a saída
                df = st.session_state.df  # Torna o df disponível no escopo local
                output_buffer = io.StringIO()
                
                try:
                    # Se for um plot, ele será capturado
                    if "plt.show()" in generated_code:
                        fig, ax = plt.subplots()
                        # Usamos um truque para que o 'exec' desenhe no nosso 'ax'
                        exec(generated_code.replace("plt.show()", ""), {"df": df, "plt": plt, "ax": ax})
                        st.pyplot(fig)
                        plt.close(fig) # Limpa a figura da memória
                        st.session_state.history.append({"role": "assistant", "image": fig, "content": "Aqui está o gráfico que você pediu."})

                    # Se for um cálculo com print()
                    else:
                        with contextlib.redirect_stdout(output_buffer):
                            exec(generated_code, {"df": df})
                        
                        text_output = output_buffer.getvalue().strip()
                        
                        # 3. Traduzir a saída se necessário
                        final_text = f"O resultado é: **{text_output}**"
                        # (A camada de tradução pode ser adicionada aqui se a saída for em inglês)

                        st.markdown(final_text)
                        st.session_state.history.append({"role": "assistant", "content": final_text})

                except Exception as e:
                    error_message = f"Ocorreu um erro ao executar o código gerado: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "content": error_message})
