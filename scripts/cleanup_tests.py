#!/usr/bin/env python3
import sys
from pathlib import Path

# Connect to your src module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.vectordb.vector_store import VectorStore

def cleanup_watcher_files():
    print("Connecting to database...")
    vs = VectorStore(database_path=settings.database_path)
    
    all_docs = vs.list_ingested_documents()
    deleted_count = 0
    
    for doc in all_docs:
        # Target ONLY the files uploaded during today's Streamlit/Watcher testing
        if doc.get("source") == "watcher_upload":
            doc_id = doc["document_id"]
            filename = doc["filename"]
            print(f"Deleting '{filename}' (ID: {doc_id})...")
            
            vs.delete_document(doc_id)
            deleted_count += 1
            
    if deleted_count == 0:
        print("No testing files found.")
    else:
        print(f"✅ Successfully cleaned up {deleted_count} testing file(s)!")

if __name__ == "__main__":
    cleanup_watcher_files()