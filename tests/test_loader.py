from rag.loader import cargar_documentos


documentos = cargar_documentos()

print("\n--- PRIMER DOCUMENTO ---")
print(documentos[0].page_content[:1000])

print("\n--- METADATOS ---")
print(documentos[0].metadata)