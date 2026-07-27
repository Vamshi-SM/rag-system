from src.vectordb.vector_store import VectorStore

# Initialize the store (it automatically uses PostgreSQL via your config)
store = VectorStore("dummy.db")

print("Clearing database...")
store.delete_all()
print("Database cleared. Current chunk count:", store.count())