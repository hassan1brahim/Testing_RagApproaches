# =============================================================================
# load.py — document ingestion + indexing for the RAG pipeline
# =============================================================================
#
# What this file does (high level):
# - Loads PDFs from a folder
# - Splits them into smaller chunks
# - Turns chunks into embeddings
# - Stores everything in ChromaDB
#
# This is the *indexing* step of RAG and must be run BEFORE querying.
# Basically: we’re building the searchable knowledge base.
#
# RAG quick overview:
# Instead of the LLM guessing from training data, we:
# 1) retrieve relevant chunks from our docs
# 2) pass them into the prompt
# → better, grounded answers
#
# Two phases:
# 1. Indexing (this file): load → chunk → embed → store
# 2. Querying (query.py): embed query → retrieve → generate response
#
# Why chunking matters:
# - LLMs have context limits
# - Embeddings work better on smaller text
# - Smaller chunks = more precise retrieval
# - Overlap helps avoid losing context at boundaries
#
# Semantic search vs keyword search:
# - Keyword search matches words
# - Semantic search matches meaning
#   (e.g. “ocean” can match “sea”, “marine”, etc.)
# =============================================================================

import argparse
import os
import shutil

# LangChain utilities for loading + splitting documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Vector DB
from langchain_chroma import Chroma

# Custom embedding function
from embed import get_embedding_function


# =============================================================================
# CONFIG
# =============================================================================

# Where ChromaDB lives on disk
CHROMA_PATH = "chroma"

# Folder containing PDFs to index
DATA_PATH = "pdfs"

# =============================================================================
# Notes on more advanced RAG ideas (future improvements)
# =============================================================================
#
# These aren’t implemented here, but good to keep in mind:
#
# 1) Query rewriting
#    - Rewrite / expand user queries before retrieval
#    - Example: “What is AI?” → “Define artificial intelligence and ML”
#    - Methods: HyDE, query expansion
#
# 2) Smarter chunking
#    - Semantic or section-based splits
#    - Parent/child chunk relationships
#    - Different chunk sizes for different content
#
# 3) Better indexing
#    - Hybrid search (keywords + vectors)
#    - Multiple embeddings per document
#    - Hierarchical indexes (summaries + details)
# =============================================================================


def main():
    """
    Entry point for the indexing pipeline.

    Flow:
    - Parse CLI args
    - Optionally reset the DB
    - Load PDFs
    - Split into chunks
    - Store embeddings in Chroma

    Usage:
        python load.py
        python load.py --reset
    """

    # CLI argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # If --reset is passed, wipe the DB first
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Load PDFs → Document objects (one per page)
    documents = load_documents()

    # Split pages into smaller chunks
    chunks = split_documents(documents)

    # Store chunks + embeddings in ChromaDB
    add_to_chroma(chunks)


def load_documents():
    """
    Loads all PDFs from DATA_PATH.

    Notes:
    - One Document per PDF page
    - Metadata includes file path + page number
    - Uses the PDF text layer (won’t work on scanned images)
    """

    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def split_documents(documents: list[Document]):
    """
    Breaks documents into smaller chunks.

    Why:
    - Smaller chunks = better retrieval precision
    - Fits LLM context limits
    - Better embedding quality

    How the recursive splitter works:
    - Tries paragraphs first
    - Then lines, sentences, words, characters
    - Keeps related text together as much as possible

    Chunk settings:
    - ~800 chars per chunk
    - 80 char overlap (~10%)
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    return splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    """
    Adds chunks to the ChromaDB vector store.

    Steps:
    - Initialize Chroma
    - Assign deterministic IDs to chunks
    - Skip chunks already indexed
    - Embed + store new chunks only

    This avoids duplicate data and keeps re-runs fast.
    """

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    # Add unique IDs to each chunk
    chunks = calculate_chunk_ids(chunks)

    # Get existing IDs for deduplication
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Keep only chunks that aren’t already indexed
    new_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")

        ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=ids)
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks: list[Document]):
    """
    Assigns stable, readable IDs to chunks.

    ID format:
        source_file:page_number:chunk_index

    Example:
        pdfs/doc.pdf:3:1
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # This was a clever way because it resets if current_page_id is different 
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database():
    """
    Deletes the entire ChromaDB directory.

    Use this when:
    - Changing embedding models
    - Changing chunk sizes
    - Rebuilding the index from scratch

    This is destructive — everything will need to be re-indexed.
    """

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    main()
