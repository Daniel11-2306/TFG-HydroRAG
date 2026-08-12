from langchain_huggingface import HuggingFaceEmbeddings


def crear_embeddings():
    """
    Crea el modelo de embeddings que utilizará el RAG.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings