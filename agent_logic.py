import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_agents import create_pandas_dataframe_agent

# Define o modelo de linguagem que será usado pelo agente
MODELO_LLM = "gemini-1.5-pro-latest"

def criar_agente_pandas(df: pd.DataFrame, api_key: str):
    """
    Inicializa e retorna um agente LangChain para análise de DataFrames pandas.

    Args:
        df (pd.DataFrame): O DataFrame que o agente irá analisar.
        api_key (str): A chave de API do Google para autenticação.

    Returns:
        Um agente LangChain configurado.
    """
    # 1. Inicializa o modelo de linguagem (LLM)
    llm = ChatGoogleGenerativeAI(
        model=MODELO_LLM,
        temperature=0,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )

    # 2. Cria a instância do Agente Pandas
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )
    
    return agent

def criar_prompt_detalhado(pergunta: str, nome_arquivo: str) -> dict:
    """
    Cria um prompt estruturado e detalhado para guiar o agente.

    Args:
        pergunta (str): A pergunta do usuário.
        nome_arquivo (str): O nome do arquivo CSV para dar contexto ao agente.

    Returns:
        dict: Um dicionário contendo o prompt formatado para a função invoke do agente.
    """
    prompt_template = f"""
    ### INSTRUÇÕES PARA O AGENTE ###

    **Sua Persona:** Você é "Agente Gemini", um analista de dados sênior, especialista em Python e Pandas. Sua comunicação é clara, objetiva e sempre em português do Brasil.

    **Seu Objetivo:** Responder à pergunta do usuário usando o DataFrame fornecido, que foi carregado do arquivo `{nome_arquivo}`.

    **Processo de Análise (Passo a Passo Obrigatório):**
    1.  **Compreensão:** Analise a estrutura do DataFrame (colunas, tipos de dados, valores ausentes) para entender os dados com os quais está trabalhando.
    2.  **Planejamento:** Antes de escrever qualquer código, descreva em palavras o plano de ação que você seguirá para responder à pergunta.
    3.  **Execução:** Escreva e execute o código Python/Pandas necessário para realizar a análise. Use o `print()` para exibir os resultados intermediários, se necessário.
    4.  **Conclusão:** Com base nos resultados do código, formule uma resposta final clara e concisa para o usuário. A resposta deve ser direta e fácil de entender.

    **Pergunta do Usuário:**
    "{pergunta}"

    **Formato da Resposta Final:**
    Apresente apenas a conclusão final, sem os pensamentos ou o código.
    """
    
    return {"input": prompt_template}
