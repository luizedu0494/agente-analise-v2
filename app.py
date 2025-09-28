# app.py - Versão Definitiva com Histórico Persistente e UI Limpa

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    st.header("Configuração")
    uploaded_file = st.file_uploader("1. Carregue seu arquivo CSV", type="csv")

    if st.button("Resetar Sessão"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.write("Desenvolvido com a sua colaboração.")
    st.sidebar.write("Foco: 100% na interação com o agente.")

# --- Lógica de Inicialização ---
if uploaded_file is not None and st.session_state.df is None:
    with st.spinner("Carregando arquivo e inicializando o agente..."):
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            groq_api_key = st.secrets["GROQ_API_KEY"]
            st.session_state.llm = ChatGroq(temperature=0, model_name="gemma2-9b-it", groq_api_key=groq_api_key)
            st.session_state.history.append({
                "role": "assistant",
                "type": "text",
                "content": f"Arquivo `{uploaded_file.name}` carregado com sucesso! O dataframe está na variável `df`. O que gostaria de saber?"
            })
        except Exception as e:
            st.error(f"Erro na inicialização: {e}")

# --- Interface Principal ---
if st.session_state.df is None:
    st.info("👆 Para começar, carregue um arquivo CSV na barra lateral.")
else:
    st.header("Converse com seus Dados")
    df = st.session_state.df # Garante que o df esteja disponível para o loop de exibição

    # --- INÍCIO DA CORREÇÃO DO HISTÓRICO PERSISTENTE ---
    # Este loop agora re-renderiza todo o histórico, incluindo gráficos e dataframes
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "code_output":
                st.markdown("**Resultado:**")
                # Se a saída for um dataframe (como de 'describe'), usa st.dataframe
                if isinstance(message["content"], pd.DataFrame):
                    st.dataframe(message["content"])
                else:
                    st.code(message["content"], language=None)
            elif message["type"] == "plot":
                # Re-executa o código do gráfico para redesenhá-lo
                fig, ax = plt.subplots()
                exec(message["content"].replace("plt.show()", ""), {"df": df, "plt": plt, "ax": ax})
                st.pyplot(fig)
                plt.close(fig)
    # --- FIM DA CORREÇÃO DO HISTÓRICO PERSISTENTE ---

    if user_prompt := st.chat_input("Ex: 'Qual a média da coluna X?' ou 'Gere um histograma para Y'"):
        st.session_state.history.append({"role": "user", "type": "text", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando sua pergunta e gerando o código..."):
                formatted_history = ""
                for msg in st.session_state.history:
                    role = "Usuário" if msg["role"] == "user" else "Assistente"
                    content = msg['content']
                    if isinstance(content, pd.DataFrame):
                        content = content.to_string() # Converte df para string para o prompt
                    formatted_history += f"{role}: {content}\n"

                code_generation_prompt = f"""
                Você é um especialista em Python e pandas. O dataframe está na variável `df`.
                Baseado no histórico da conversa, gere o código Python para responder à última pergunta do usuário.
                - Para cálculos e descrições (mean, dtypes, describe), **SEMPRE** use `print()`.
                - Para gráficos, use `plt.show()`.
                - Forneça apenas o bloco de código Python.

                ### Histórico da Conversa ###
                {formatted_history}
                """
                
                code_response = st.session_state.llm.invoke(code_generation_prompt)
                generated_code = code_response.content.strip().replace("```python", "").replace("```", "").strip()
                
                if "plt.show()" not in generated_code and "print(" not in generated_code:
                    generated_code = f"print({generated_code})"

                st.write("Código gerado:")
                st.code(generated_code)

            with st.spinner("Executando código e preparando a resposta..."):
                try:
                    if "plt.show()" in generated_code:
                        # Salva o código do gráfico no histórico para ser re-renderizado
                        st.session_state.history.append({"role": "assistant", "type": "plot", "content": generated_code})
                        # Executa para exibir pela primeira vez
                        fig, ax = plt.subplots()
                        exec(generated_code.replace("plt.show()", ""), {"df": df, "plt": plt, "ax": ax})
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        # --- INÍCIO DA CORREÇÃO DA SAÍDA DE TEXTO ---
                        # Usamos exec para avaliar o código e capturar o resultado
                        # Isso nos permite verificar se o resultado é um dataframe
                        result = None
                        with contextlib.redirect_stdout(io.StringIO()) as stdout:
                            # O 'result' será o que a última linha do código retorna
                            result = eval(generated_code.replace("print(", "").replace(")", ""), {"df": df})
                        
                        # Se o resultado for um dataframe, o salvamos como tal
                        if isinstance(result, (pd.DataFrame, pd.Series)):
                            st.session_state.history.append({"role": "assistant", "type": "code_output", "content": result})
                            st.markdown("**Resultado:**")
                            st.dataframe(result)
                        else: # Caso contrário, tratamos como texto
                            text_output = str(result)
                            st.session_state.history.append({"role": "assistant", "type": "code_output", "content": text_output})
                            st.markdown("**Resultado:**")
                            st.code(text_output, language=None)
                        # --- FIM DA CORREÇÃO DA SAÍDA DE TEXTO ---

                except Exception as e:
                    error_message = f"Ocorreu um erro ao executar o código: {e}"
                    st.error(error_message)
                    st.session_state.history.append({"role": "assistant", "type": "text", "content": f"Erro: {error_message}"})
