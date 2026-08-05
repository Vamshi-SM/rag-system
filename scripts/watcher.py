#!/usr/bin/env python3
"""Backend File Watcher for Automated Document Ingestion."""

import os
import sys
import time
import uuid
import traceback
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Hardcode credentials to ensure Google Vertex AI authentication works in background process
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Vamsi/Downloads/rag-system-503214-e7748adb1ae1.json"

# Ensure the src module can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.google_embedding import GoogleEmbedding
from src.vectordb.vector_store import VectorStore

# Define the Drop Zone
UPLOAD_FOLDER = Path("C:/Users/Vamsi/Downloads/rag-system-final/rag-system/data/upload_files")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


class VectorIngestionHandler(FileSystemEventHandler):
    """Listens for new files and automatically ingests them into the Vector DB."""

    def __init__(self, vector_store: VectorStore, embedder: GoogleEmbedding):
        super().__init__()
        self.vector_store = vector_store
        self.embedder = embedder

    def on_created(self, event):
        """Triggered exactly when a new file is dropped into the folder."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        print(f"\n[TRIGGER] New file detected: {file_path.name}")

        # Slight delay to ensure OS finishes writing the file before reading
        time.sleep(1.5)

        self.process_and_ingest(file_path)

    def process_and_ingest(self, file_path: Path):
        """Reads, chunks, embeds, and stores the file."""
        import hashlib
        try:
            # 1. Read the file
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                print(f"[SKIP] {file_path.name} is empty.")
                return
                
            # 2. Calculate real checksum (MD5 hash of the content)
            file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            
            # 3. Check if document is already in the database
            existing_doc_id = self.vector_store.is_document_ingested(file_hash)
            if existing_doc_id:
                print(f"⏩ [SKIP] '{file_path.name}' is already in the database (Doc ID: {existing_doc_id}).")
                return

            print(f"[PROCESSING] Chunking {file_path.name}...")
            # 4. Chunking
            chunk_size = 1000
            raw_chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

            # 5. Format with top-level keys
            document_id = str(uuid.uuid4())
            formatted_chunks = []
            for i, text in enumerate(raw_chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                formatted_chunks.append({
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": text,
                    "metadata": {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "source": file_path.name
                    }
                })

            print(f"[EMBEDDING] Generating vectors for {len(formatted_chunks)} chunks via Vertex AI...")
            # 6. Generate Embeddings
            embedded_chunks = self.embedder.embed_documents(formatted_chunks)

            print(f"[DATABASE] Storing vectors in PostgreSQL/pgvector...")
            # 7. Save to Vector Store
            self.vector_store.insert_many(embedded_chunks)

            # 8. Register Document using the real hash
            self.vector_store.register_document(
                document_id=document_id,
                checksum=file_hash,
                filename=file_path.name,
                source="watcher_upload",
                chunk_count=len(formatted_chunks)
            )

            print(f"✅ [SUCCESS] {file_path.name} is now available for RAG queries!\n")

        except Exception as e:
            print(f"❌ [ERROR] Failed to ingest {file_path.name}: {str(e)}")
            traceback.print_exc()
            
        """Reads, chunks, embeds, and stores the file."""
        try:
            # 1. Read the file
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                print(f"[SKIP] {file_path.name} is empty.")
                return

            print(f"[PROCESSING] Chunking {file_path.name}...")
            # 2. Chunking
            chunk_size = 1000
            raw_chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

            # 3. Format with top-level keys expected by GoogleEmbedding
            document_id = str(uuid.uuid4())
            formatted_chunks = []
            for i, text in enumerate(raw_chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                formatted_chunks.append({
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": text,
                    "metadata": {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "source": file_path.name
                    }
                })

            print(f"[EMBEDDING] Generating vectors for {len(formatted_chunks)} chunks via Vertex AI...")
            # 4. Generate Embeddings
            embedded_chunks = self.embedder.embed_documents(formatted_chunks)

            print(f"[DATABASE] Storing vectors in PostgreSQL/pgvector...")
            # 5. Save to Vector Store
            self.vector_store.insert_many(embedded_chunks)

            # 6. Register Document
            self.vector_store.register_document(
                document_id=document_id,
                checksum=document_id,
                filename=file_path.name,
                source="watcher_upload",
                chunk_count=len(formatted_chunks)
            )

            print(f"✅ [SUCCESS] {file_path.name} is now available for RAG queries!\n")

        except Exception as e:
            print(f"❌ [ERROR] Failed to ingest {file_path.name}: {str(e)}")
            traceback.print_exc()


def start_watcher():
    """Initializes the infrastructure and starts the watchdog observer."""
    print("Initializing Database and Embedding Models...")

    embedder = GoogleEmbedding(
        project_id=settings.gcp_project_id,
        location=settings.gcp_location,
        model=settings.embedding_model,
    )
    vector_store = VectorStore(
        database_path=settings.database_path,
        default_top_k=settings.top_k
    )

    event_handler = VectorIngestionHandler(vector_store, embedder)
    observer = Observer()
    observer.schedule(event_handler, str(UPLOAD_FOLDER), recursive=False)

    observer.start()
    print(f"\n👀 Watcher active. Listening for new files in: {UPLOAD_FOLDER}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watcher()