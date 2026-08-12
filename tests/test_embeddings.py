from rag.embeddings import crear_embeddings


embeddings = crear_embeddings()

texto = "El acuífero presenta variaciones en los niveles freáticos."

vector = embeddings.embed_query(texto)

print("Embedding generado correctamente")
print(f"Dimensión del vector: {len(vector)}")
print(f"Primeros valores: {vector[:5]}")