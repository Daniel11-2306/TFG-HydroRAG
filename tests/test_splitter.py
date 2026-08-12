from rag.loader import cargar_documentos
from rag.splitter import dividir_documentos


documentos = cargar_documentos()

chunks = dividir_documentos(documentos)

print("\n--- PRIMER CHUNK ---")
print(chunks[0].page_content)

print("\n--- METADATOS ---")
print(chunks[0].metadata)