from rag.loader import cargar_documentos
from rag.splitter import dividir_documentos
from rag.vectorstore import crear_vectorstore
from rag.generator import generar_respuesta


def main():

    print("===================================")
    print("       PROTOTIPO RAG")
    print("===================================")

    # 1. Cargar documentos
    print("\n[1] Cargando documentos...")
    documentos = cargar_documentos()

    # 2. Crear chunks
    print("[2] Dividiendo documentos...")
    chunks = dividir_documentos(documentos)

    print(f"Chunks generados: {len(chunks)}")

    # 3. Crear vectorstore
    print("[3] Creando base vectorial...")
    vectorstore = crear_vectorstore(chunks)

    # 4. Pregunta
    pregunta = input(
        "\nEscribe una pregunta sobre el documento:\n> "
    )

    # 5. Recuperación
    print("\n[4] Buscando información relevante...")

    documentos_relevantes = vectorstore.similarity_search(
        pregunta,
        k=5
    )

    print(
        f"Documentos recuperados: "
        f"{len(documentos_relevantes)}"
    )

    # 6. Generación
    print("\n[5] Generando respuesta...")

    respuesta = generar_respuesta(
        pregunta,
        documentos_relevantes
    )

    print("\n===================================")
    print("RESPUESTA")
    print("===================================\n")

    print(respuesta)


if __name__ == "__main__":
    main()