#!/usr/bin/env python3
import sys
from pathlib import Path

# Connect to your src module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.vectordb.vector_store import VectorStore
from src.vectordb.schema import CHUNKS_TABLE, VEC_TABLE, CHUNKS_FTS_TABLE

def clean_orphans():
    print("Connecting to database to hunt for ghost chunks...")
    vs = VectorStore(database_path=settings.database_path)
    
    if vs.is_postgres:
        # PostgreSQL targeted deletion
        rows = vs.database.query(f"DELETE FROM {CHUNKS_TABLE} WHERE document_id IS NULL RETURNING id")
        print(f"✅ Obliterated {len(rows)} orphaned chunks!")
    else:
        # SQLite targeted deletion
        rows = vs.database.query(f"SELECT rowid FROM {CHUNKS_TABLE} WHERE document_id IS NULL")
        rowids = [row["rowid"] for row in rows]
        
        if not rowids:
            print("No orphaned chunks found. You are completely clean!")
            return
            
        placeholders = ",".join("?" * len(rowids))
        vs.database.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE rowid IN ({placeholders})", tuple(rowids))
        
        if vs._dimensions is not None:
            vs.database.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid IN ({placeholders})", tuple(rowids))
            
        if vs.database.fts_available:
            vs.database.execute(f"DELETE FROM {CHUNKS_FTS_TABLE} WHERE rowid IN ({placeholders})", tuple(rowids))
            
        print(f"✅ Obliterated {len(rowids)} orphaned ghost chunks!")

if __name__ == "__main__":
    clean_orphans()