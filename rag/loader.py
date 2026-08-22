from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def cargar_documentos(ruta_documentos: str = "data/documents"):
    """
    Carga todos los archivos PDF encontrados en la carpeta indicada.

    Devuelve:
        documentos: páginas cargadas de todos los PDFs.
        numero_documentos: cantidad de archivos PDF.
    """

    ruta = Path(ruta_documentos)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de documentos: {ruta}"
        )

    archivos_pdf = list(ruta.glob("*.pdf"))

    if not archivos_pdf:
        raise FileNotFoundError(
            f"No se encontraron archivos PDF en: {ruta}"
        )

    documentos = []

    for archivo in archivos_pdf:
        print(f"Cargando: {archivo.name}")

        loader = PyPDFLoader(str(archivo))
        paginas = loader.load()

        documentos.extend(paginas)

        print(f"  → {len(paginas)} páginas cargadas")

    numero_documentos = len(archivos_pdf)

    print(f"\nDocumentos PDF: {numero_documentos}")
    print(f"Total de páginas cargadas: {len(documentos)}")

    return documentos, numero_documentos