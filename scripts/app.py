#!/usr/bin/env python3
"""Streamlit Frontend for Enterprise RAG System."""

import streamlit as st
import os
import time
from pathlib import Path
import sys

# Ensure the src module can be found for the RAG pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings
from src.llm.gemini_chat import GeminiChat
from src.embeddings.google_embedding import GoogleEmbedding
from src.retrieval.retriever import Retriever
from src.vectordb.vector_store import VectorStore
from src.rag.rag_pipeline import RAGPipeline

# Define the Drop Zone exactly as configured in the watcher
UPLOAD_FOLDER = Path("C:/Users/Vamsi/Downloads/rag-system-final/rag-system/data/upload_files")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Initialize RAG Pipeline (Cached so it doesn't reload on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_pipeline():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Vamsi/Downloads/rag-system-503214-e7748adb1ae1.json"
    
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
    )
    llm = GeminiChat(
        project_id=settings.gcp_project_id,
        location=settings.gcp_location,
    )
    return RAGPipeline(retriever=retriever, llm=llm)

pipeline = get_rag_pipeline()

# ---------------------------------------------------------------------------
# UI Setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖")
st.title("💬 Enterprise RAG Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize processed files tracker to avoid re-uploading the same file
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ---------------------------------------------------------------------------
# Sidebar: File Upload handling
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    st.write("Drop files here. The backend watcher will automatically ingest them.")
    
    uploaded_files = st.file_uploader("Upload TXT or MD files", accept_multiple_files=True, type=["txt", "md"])

    if uploaded_files:
        new_files_detected = False
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                new_files_detected = True
                break
        
        if new_files_detected:
            # Display processing notification
            with st.spinner("Processing... saving and triggering backend watcher..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_files:
                        # 1. Save file to the upload_files folder
                        file_path = UPLOAD_FOLDER / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Mark as processed in the frontend session
                        st.session_state.processed_files.add(uploaded_file.name)
                
                # 2. Simulate wait time for the background watcher script to finish embedding
                time.sleep(4) 
                
            # 3. Update UI notification
            st.success("✅ Embedding done! You can now ask questions.")

# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Query the RAG Pipeline
                response = pipeline.answer(prompt)
                
                answer_text = response.answer
                if response.sources:
                    answer_text += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in response.sources])
                
                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                st.error(f"Error querying the database: {str(e)}")