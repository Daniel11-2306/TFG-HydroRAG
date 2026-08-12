import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from rag.loader import cargar_documentos
from rag.splitter import dividir_documentos
from rag.vectorstore import crear_vectorstore
from rag.generator import generar_respuesta

# --------------------------------------------------
# Configuración
# --------------------------------------------------

st.set_page_config(
    page_title="HydroRAG",
    page_icon=" ",
    layout="wide"
)


# --------------------------------------------------
# Título
# --------------------------------------------------

st.title("HydroRAG-prototipo 1")

st.write(
    "Sistema de recuperación aumentada por generación "
    "para consulta de información hidrogeológica."
)


# --------------------------------------------------
# Inicialización del RAG
# --------------------------------------------------

@st.cache_resource
def inicializar_rag():

    documentos = cargar_documentos()

    chunks = dividir_documentos(documentos)

    vectorstore = crear_vectorstore(chunks)

    return vectorstore


# --------------------------------------------------
# Crear RAG
# --------------------------------------------------

with st.spinner("Inicializando HydroRAG..."):

    vectorstore = inicializar_rag()


st.success("Sistema listo para realizar consultas.")


# --------------------------------------------------
# Consulta
# --------------------------------------------------

pregunta = st.text_input(
    "Realiza una consulta hidrogeológica:",
    placeholder="Ejemplo: ¿Cuál es el balance de aguas subterráneas?"
)


# --------------------------------------------------
# Procesamiento
# --------------------------------------------------

if pregunta:

    with st.spinner("Buscando información relevante..."):

        documentos_relevantes = vectorstore.similarity_search(
            pregunta,
            k=5
        )

    with st.spinner("Generando respuesta..."):

        respuesta = generar_respuesta(
            pregunta,
            documentos_relevantes
        )

    st.subheader("Respuesta")

    st.write(respuesta)


    # --------------------------------------------------
    # Fuentes
    # --------------------------------------------------

    st.subheader("Fuentes recuperadas")

    for i, documento in enumerate(
        documentos_relevantes,
        start=1
    ):

        with st.expander(
            f"Fuente {i}"
        ):

            st.write(
                documento.page_content
            )

            st.caption(
                f"Página: "
                f"{documento.metadata.get('page', 'N/A')}"
            )