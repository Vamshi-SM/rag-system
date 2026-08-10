#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Connect to your src module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.vectordb.vector_store import VectorStore
from src.vectordb.schema import CHUNKS_TABLE

def scrub_corrupted_metadata():
    print("Scanning database for chunks with missing metadata...")
    vs = VectorStore(database_path=settings.database_path)
    
    # Fetch all chunks to inspect their metadata
    rows = vs.database.query(f"SELECT id, document_id, metadata FROM {CHUNKS_TABLE}")
    
    bad_document_ids = set()
    
    for row in rows:
        # Parse metadata depending on if it's Postgres (dict) or SQLite (JSON string)
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
            
        # If the chunk doesn't know where it came from, tag the document for deletion
        if "filename" not in meta and "source" not in meta:
            if row["document_id"]:
                bad_document_ids.add(row["document_id"])

    if not bad_document_ids:
        print("✅ No corrupted metadata found! Your database is clean.")
        return

    print(f"Found {len(bad_document_ids)} document(s) with missing source metadata. Purging them now...")
    
    deleted_total = 0
    for doc_id in bad_document_ids:
        chunks_removed = vs.delete_document(doc_id)
        deleted_total += chunks_removed
        print(f" - Purged Document ID: {doc_id} ({chunks_removed} chunks)")
        
    print(f"✅ Successfully scrubbed {deleted_total} bad chunks from the database.")

if __name__ == "__main__":
    scrub_corrupted_metadata()