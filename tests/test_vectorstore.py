from rag.loader import cargar_documentos
from rag.splitter import dividir_documentos
from rag.vectorstore import crear_vectorstore


def main():

    print("1. Cargando documentos...")
    documentos = cargar_documentos()

    print("2. Dividiendo documentos...")
    chunks = dividir_documentos(documentos)

    print(f"Chunks: {len(chunks)}")

    print("3. Creando base vectorial...")
    vectorstore = crear_vectorstore(chunks)

    print("4. Base vectorial creada correctamente.")

    resultados = vectorstore.similarity_search(
        "¿Cuál es el balance de agua subterránea del Valle del Mezquital?",
        k=3
    )

    print("\n--- RESULTADOS ---")

    for i, resultado in enumerate(resultados, start=1):
        print(f"\nResultado {i}:")
        print(resultado.page_content[:1000])


if __name__ == "__main__":
    main()