from src.vectordb.vector_store import VectorStore

store = VectorStore("dummy.db")   # ignored when DATABASE_BACKEND=postgres

embedding = [0.1] * 768

# 1. Insert the chunk
store.insert_many([
    {
        "text": "This is a PostgreSQL test chunk.",
        "embedding": embedding,
        "metadata": {
            "id": "row-1",
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
        },
    }
])
print("Inserted successfully!")

# 2. Test similarity search
results = store.similarity_search(
    embedding,
    top_k=1,
)

print("\nSearch Results:")
# Using json.dumps to print it nicely formatted
import json
print(json.dumps(results, indent=2))

# ---------------------------------------------------------
# Step 4: Test delete_document
# ---------------------------------------------------------
print("\n--- Testing delete_document ---")
deleted = store.delete_document("doc-1")
print("Deleted count:", deleted)
print("Total chunks remaining:", store.count())

# ---------------------------------------------------------
# Step 6: Test document registry (Doing this before delete_all)
# ---------------------------------------------------------
print("\n--- Testing document registry ---")
store.register_document(
    document_id="doc-100",
    checksum="abc123",
    filename="test.txt",
    source="local",
    chunk_count=5,
)
print("Ingested documents list:", store.list_ingested_documents())
print("Lookup by checksum 'abc123':", store.is_document_ingested("abc123"))

# ---------------------------------------------------------
# Step 5: Test delete_all
# ---------------------------------------------------------
print("\n--- Testing delete_all ---")
store.delete_all()
print("Total chunks after delete_all:", store.count())
print("Registry after delete_all:", store.list_ingested_documents())