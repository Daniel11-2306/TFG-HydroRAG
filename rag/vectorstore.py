import os

from dotenv import load_dotenv
from langchain_postgres import PGVector

from rag.embeddings import crear_embeddings


load_dotenv()


def crear_vectorstore(chunks):
    """
    Crea el almacén vectorial en PostgreSQL + pgvector
    a partir de los chunks.
    """

    embeddings = crear_embeddings()

    connection = (
        f"postgresql+psycopg://"
        f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'hydrorag')}"
    )

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hydrorag_real",
        connection=connection,
        use_jsonb=True,
    )

    return vectorstore