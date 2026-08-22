import os

from dotenv import load_dotenv
from langchain_postgres import PGVector

from rag.embeddings import crear_embeddings


load_dotenv()


COLLECTION_NAME = "hydrorag_real"


def obtener_conexion():
    return (
        f"postgresql+psycopg://"
        f"{os.getenv('POSTGRES_USER', 'postgres')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'hydrorag')}"
    )


def crear_vectorstore(chunks):
    """
    Conecta con el almacén vectorial PostgreSQL + pgvector.
    """

    embeddings = crear_embeddings()
    connection = obtener_conexion()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
    )

    return vectorstore