from langchain_chroma import Chroma
from rag.embeddings import crear_embeddings


def crear_vectorstore(chunks):
    """
    Crea una base vectorial a partir de los chunks
    utilizando embeddings.
    """

    embeddings = crear_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data/vector_db"
    )

    return vectorstore