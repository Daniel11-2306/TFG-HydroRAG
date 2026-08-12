from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def crear_llm():
    """
    Crea el modelo de lenguaje que utilizará el RAG.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    return llm


def generar_respuesta(pregunta, documentos):

    llm = crear_llm()

    contexto = "\n\n".join(
        documento.page_content
        for documento in documentos
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Eres un asistente especializado en información hidrogeológica.

        Responde la pregunta utilizando únicamente la información
        proporcionada en el contexto.

        Si la información necesaria no aparece en el contexto,
        indica claramente que no se encuentra en los documentos
        disponibles.

        No inventes datos, referencias ni valores.

        CONTEXTO:
        {contexto}

        PREGUNTA:
        {pregunta}

        RESPUESTA:
        """
    )

    mensajes = prompt.format_messages(
        contexto=contexto,
        pregunta=pregunta
    )

    respuesta = llm.invoke(mensajes)

    return respuesta.content