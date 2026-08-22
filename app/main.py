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


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="HydroRAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       CONFIGURACIÓN GENERAL
       ====================================================== */

    .stApp {
        background-color: #f5f7fa;
        color: #1f2937;
    }

    /* Texto general de Streamlit */
    .stMarkdown,
    .stTextInput,
    .stTextArea,
    .stCaption,
    .stMetric,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #1f2937;
    }

    /* Títulos */
    h1, h2, h3, h4 {
        color: #17324d !important;
    }


    /* ======================================================
       CABECERA
       ====================================================== */

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f4c5c,
            #197278
        );
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }

    .hero h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .hero p {
        color: white;
        margin-top: 0.6rem;
        margin-bottom: 0;
        font-size: 1.05rem;
        opacity: 0.92;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background-color: #eef3f6;
    }

    .sidebar-card {
        background-color: white;
        color: #1f2937;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #dce3e8;
        margin-bottom: 1rem;
    }

    .sidebar-card strong {
        color: #17324d;
    }


    /* ======================================================
       MÉTRICAS
       ====================================================== */

    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #dce3e8;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #17324d !important;
        font-weight: 700;
    }


    /* ======================================================
       CAJA DE CONSULTA
       ====================================================== */

    [data-testid="stTextArea"] textarea {
        background-color: white;
        color: #1f2937;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #94a3b8;
    }


    /* ======================================================
       BOTÓN
       ====================================================== */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 3rem;
    }


    /* ======================================================
       TARJETA DE RESPUESTA
       ====================================================== */

    .response-card {
        background-color: white;
        color: #1f2937;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #dce3e8;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }


    /* ======================================================
       FUENTES
       ====================================================== */

    [data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #dce3e8;
        border-radius: 12px;
    }


    /* ======================================================
       TEXTO SECUNDARIO
       ====================================================== */

    .small-text {
        color: #64748b;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1> HydroRAG</h1>
        <p>
            Sistema de recuperación aumentada por generación
            para consulta de información hidrogeológica.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Sistema")

    st.markdown(
        """
        <div class="sidebar-card">
            <strong>Estado del sistema</strong><br><br>
            🟢 HydroRAG operativo
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Arquitectura")

    st.write("PostgreSQL")
    st.write("pgvector")
    st.write("HuggingFace Embeddings")
    st.write("OpenAI")

    st.divider()

    st.subheader("Sobre HydroRAG")

    st.caption(
        "HydroRAG utiliza recuperación semántica "
        "para localizar información relevante en "
        "documentos hidrogeológicos y generar "
        "respuestas basadas en el contexto recuperado."
    )


# ============================================================
# INICIALIZACIÓN DEL RAG
# ============================================================

@st.cache_resource
def inicializar_rag():

    documentos, numero_documentos = cargar_documentos()

    chunks = dividir_documentos(documentos)

    vectorstore = crear_vectorstore(chunks)

    numero_paginas = len(documentos)
    numero_chunks = len(chunks)

    return (
        vectorstore,
        numero_documentos,
        numero_paginas,
        numero_chunks
    )


with st.spinner("Inicializando HydroRAG..."):

    (
        vectorstore,
        numero_documentos,
        numero_paginas,
        numero_chunks
    ) = inicializar_rag()

# ============================================================
# INFORMACIÓN DEL SISTEMA
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Documentos",
        numero_documentos
    )

with col2:
    st.metric(
        "Páginas",
        numero_paginas
    )

with col3:
    st.metric(
        "Chunks",
        numero_chunks
    )

st.write("")
st.divider()


# ============================================================
# CONSULTA
# ============================================================

st.subheader("Consulta hidrogeológica")

st.write(
    "Introduce una pregunta sobre el contenido de los "
    "documentos disponibles."
)


pregunta = st.text_area(
    "Pregunta",
    placeholder=(
        "Ejemplo: ¿Cuál es el balance de aguas "
        "subterráneas del acuífero del Valle del Mezquital?"
    ),
    height=110,
    label_visibility="collapsed",
)


consultar = st.button(
    "Consultar HydroRAG",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESAMIENTO
# ============================================================

if consultar:

    if not pregunta.strip():

        st.warning(
            "Introduce una pregunta antes de realizar la consulta."
        )

    else:

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


        # ====================================================
        # RESPUESTA
        # ====================================================

        st.subheader("Respuesta")

        st.markdown(
            f"""
            <div class="response-card">
                {respuesta}
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # FUENTES
        # ====================================================

        st.subheader("Fuentes recuperadas")

        st.caption(
            f"HydroRAG recuperó "
            f"{len(documentos_relevantes)} fragmentos "
            f"relevantes para esta consulta."
        )


        for i, documento in enumerate(
            documentos_relevantes,
            start=1
        ):

            pagina = documento.metadata.get(
                "page",
                "N/A"
            )

            fuente = documento.metadata.get(
                "source",
                "Documento no especificado"
            )

            nombre_fuente = Path(fuente).name


            with st.expander(
                f"Fuente {i} · Página {pagina}"
            ):

                st.markdown(
                    f"**Documento:** `{nombre_fuente}`"
                )

                st.divider()

                st.write(
                    documento.page_content
                )