from langchain_text_splitters import RecursiveCharacterTextSplitter


def dividir_documentos(documentos):
    """
    Divide los documentos en fragmentos pequeños
    manteniendo el contexto entre ellos.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(documentos)

    print(f"Documentos originales: {len(documentos)}")
    print(f"Chunks generados: {len(chunks)}")

    return chunks