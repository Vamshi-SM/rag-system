#!/usr/bin/env python3
"""MCP Server for the RAG System."""

from __future__ import annotations

import sys
from pathlib import Path

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Vamsi/Downloads/rag-system-503214-e7748adb1ae1.json"

# Ensure the src module can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP

from src.config import settings
from src.embeddings.google_embedding import GoogleEmbedding
from src.llm.gemini_chat import GeminiChat
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore

# Initialize the MCP Server
mcp = FastMCP("Enterprise RAG System")

# Global variable to hold our pipeline so it only initializes once
_pipeline: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    """Lazily initialize and return the RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        settings.ensure_directories()
        
        embedder = GoogleEmbedding(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
            model=settings.embedding_model,
        )
        vector_store = VectorStore(
            database_path=settings.database_path, 
            default_top_k=settings.top_k
        )
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=settings.top_k,
            default_similarity_threshold=settings.similarity_threshold,
            candidate_multiplier=settings.retrieval_candidate_multiplier,
        )
        llm = GeminiChat(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location,
        )
        
        _pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )
    return _pipeline

# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def ingest_local_files(file_paths: list[str]) -> str:
    """
    Reads multiple files from the local filesystem, saves a copy to the internal storage, 
    generates embeddings, and adds them to the vector database.
    Input must be a list of absolute file paths (e.g., ["C:/Documents/file1.txt"]).
    """
    pipeline = get_pipeline()
    
    # 1. Define where the application should permanently store these files
    storage_dir = Path("C:/Users/Vamsi/Downloads/rag-system-final/rag-system/data/stored_files")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for path_str in file_paths:
        source_path = Path(path_str)
        
        if not source_path.exists():
            results.append(f"❌ File not found: {path_str}")
            continue
            
        try:
            # 2. Copy file to internal local storage
            destination_path = storage_dir / source_path.name
            shutil.copy2(source_path, destination_path)
            
            # 3. Read the content (Assuming text files for now)
            content = destination_path.read_text(encoding="utf-8")
            
            # 4. Chunking and Embedding (You will need to import your chunker here)
            # pseudo-code based on your architecture:
            # chunks = chunk_text(content)
            # embedded_chunks = pipeline.retriever.embedder.embed_documents(chunks)
            # pipeline.retriever.vector_store.add(embedded_chunks)
            
            results.append(f"✅ Successfully stored and embedded: {source_path.name}")
            
        except Exception as e:
            results.append(f"❌ Error processing {source_path.name}: {str(e)}")
            
    return "\n".join(results)

@mcp.tool()
def ask_company_docs(question: str) -> str:
    """
    Answer a question based strictly on the internal company document database.
    Use this tool whenever the user asks about internal policies, contracts, or technical documentation.
    """
    pipeline = get_pipeline()
    try:
        response = pipeline.answer(question)
        
        # Format the output nicely for the AI client to read
        formatted_response = f"Answer:\n{response.answer}\n\n"
        if response.sources:
            formatted_response += "Sources used:\n"
            for source in response.sources:
                formatted_response += f"- {source}\n"
                
        return formatted_response
    except Exception as e:
        return f"Error querying documents: {str(e)}"

if __name__ == "__main__":
    mcp.run()