#!/usr/bin/env python3
import sys
from pathlib import Path

# Connect to your src module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.vectordb.vector_store import VectorStore

def delete_specific_file(target_filename: str):
    print(f"Connecting to database to find '{target_filename}'...")
    vs = VectorStore(database_path=settings.database_path)
    
    # Fetch all registered documents
    all_docs = vs.list_ingested_documents()
    
    deleted_count = 0
    for doc in all_docs:
        if doc["filename"] == target_filename:
            doc_id = doc["document_id"]
            print(f"Found match! Deleting Document ID: {doc_id}")
            
            # Use your built-in delete method to wipe the chunks and registry entry
            chunks_removed = vs.delete_document(doc_id)
            print(f"Removed {chunks_removed} chunks for this document.")
            deleted_count += 1
            
    if deleted_count == 0:
        print(f"Could not find any files named '{target_filename}' in the database.")
    else:
        print(f"✅ Successfully completely deleted {deleted_count} instance(s) of '{target_filename}'.")

if __name__ == "__main__":
    # You can change this to any filename you want to remove in the future
    delete_specific_file("rag_system.txt")