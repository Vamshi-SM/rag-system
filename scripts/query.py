#!/usr/bin/env python3
"""Interactive RAG query CLI (Phase 7 retrieval + Phase 8 generation).

Usage:
    python scripts/query.py                                 # interactive mode
    python scripts/query.py "What is our refund policy?"    # one-shot mode
    python scripts/query.py "..." --top-k 3 --threshold 0.75
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.embeddings.google_embedding import GoogleEmbedding
from src.llm.gemini_chat import GeminiChat
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger
from src.vectordb.vector_store import VectorStore

logger = get_logger(__name__)

_EXIT_COMMANDS = {"exit", "quit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question to ask. Omit to enter interactive mode.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Minimum similarity score to keep a chunk."
    )
    parser.add_argument(
        "--filename", type=str, default=None, help="Restrict retrieval to a specific filename."
    )
    return parser.parse_args()


def build_pipeline() -> RAGPipeline:
    """Wire up the Phase 7/8 pipeline from Phase 2-5 components + config."""
    embedder = GoogleEmbedding(
        project_id=settings.gcp_project_id,
        location=settings.gcp_location,
        model=settings.embedding_model,
    )
    vector_store = VectorStore(database_path=settings.database_path, default_top_k=settings.top_k)
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        default_top_k=settings.top_k,
        default_similarity_threshold=settings.similarity_threshold,
        candidate_multiplier=settings.retrieval_candidate_multiplier,
    )
    # Passed location=settings.gcp_location so it routes to your target region (e.g. asia-south1)
    llm = GeminiChat(
        project_id=settings.gcp_project_id,
        location=settings.gcp_location,
    )
    print("\n========== LLM INFO ==========")
    print("Class     :", type(llm))
    print("Module    :", llm.__class__.__module__)
    print("Model     :", llm.model)
    print("Project   :", settings.gcp_project_id)
    print("Location  :", settings.gcp_location)
    print("==============================\n")
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
    )


def execute_stream_query(pipeline: RAGPipeline, question: str, args: argparse.Namespace) -> None:
    print("Searching...")
    stream_gen, chunks, sources = pipeline.stream_answer(
        question,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        filename=args.filename,
    )
    print(f"Retrieved {len(chunks)} chunks.")
    print("Generating answer...\n")

    print("Answer\n")
    start_gen = time.perf_counter()
    for token in stream_gen:
        print(token, end="", flush=True)
    elapsed_gen = time.perf_counter() - start_gen
    print("\n")

    if sources:
        print("Sources\n")
        for source in sources:
            print(f"{source}")

    print(f"\n({elapsed_gen:.2f}s generation time, {len(chunks)} chunks used)\n")


def run_one_shot(pipeline: RAGPipeline, args: argparse.Namespace) -> None:
    execute_stream_query(pipeline, args.question, args)


def run_interactive(pipeline: RAGPipeline, args: argparse.Namespace) -> None:
    print("RAG Query Console. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("Ask a question\n\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in _EXIT_COMMANDS:
            print("Goodbye.")
            break

        execute_stream_query(pipeline, question, args)


def main() -> None:
    args = parse_args()
    settings.ensure_directories()

    start = time.time()
    pipeline = build_pipeline()

    try:
        if args.question:
            run_one_shot(pipeline, args)
        else:
            run_interactive(pipeline, args)
    finally:
        pipeline.retriever.vector_store.close()
        logger.info("Query session finished in %.2fs", time.time() - start)


if __name__ == "__main__":
    main()