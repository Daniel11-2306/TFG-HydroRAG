from pathlib import Path
import faiss
import pickle
import numpy as np

from rag.embeddings import generar_embeddings


VECTOR_DB = Path("data/vector_db")


def crear_vector_store(chunks):
    """
    Crea un índice FAISS a partir de los chunks.
    """

    textos = [chunk.page_content for chunk in chunks]

    # Generar embeddings
    embeddings = generar_embeddings(textos)

    embeddings = np.array(embeddings).astype("float32")

    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Agregar embeddings
    index.add(embeddings)

    # Crear directorio
    VECTOR_DB.mkdir(parents=True, exist_ok=True)

    # Guardar índice
    faiss.write_index(
        index,
        str(VECTOR_DB / "index.faiss")
    )

    # Guardar chunks
    with open(VECTOR_DB / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Vector store creado correctamente")
    print(f"Vectores almacenados: {index.ntotal}")
    print(f"Dimensión: {dimension}")