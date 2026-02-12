# =============================================================================
# query.py — RAG query engine (Chroma retrieval + Ollama LLM)
# =============================================================================
#
# What this file does:
# - Takes a natural language question
# - Searches ChromaDB for the most relevant chunks
# - Builds a prompt with that context
# - Sends it to a local LLM via Ollama (Mistral)
# - Prints the answer + where it came from (sources)
#
# This is the *querying* side of RAG — it relies on the index from load.py.
#
# RAG flow (quick):
# 1) embed the question (must use same embed model as indexing)
# 2) similarity search in Chroma (top-k chunks)
# 3) glue chunks into a context block
# 4) ask the LLM to answer using only that context
#
# Why not just return the chunks?
# Chunks are fragments — the LLM stitches them into an actual answer.
#
# Ollama note:
# Ollama runs models locally, so it’s private + no API fees (but you need hardware).
# =============================================================================

import argparse

# LangChain imports
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Must match the embedding function used in load.py
from embed import get_embedding_function


# =============================================================================
# CONFIG
# =============================================================================

# Must match CHROMA_PATH in load.py
CHROMA_PATH = "chroma"


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
# Simple “context + question” prompt.
# The key line is “based only on context” — helps reduce hallucinations.
# You can extend this later with stuff like:
# - “If context is missing info, say you don’t know”
# - formatting rules (bullets, citations, etc.)
# =============================================================================

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    response_text = query_rag(args.query_text)
    print(response_text)


def query_rag(query_text: str):
    """
    Runs the full RAG query:
    - connect to Chroma
    - retrieve top-k chunks
    - build prompt
    - call Ollama (Mistral)
    - print answer + sources
    """

    # Same embedding model as indexing or retrieval will be trash
    embedding_function = get_embedding_function()

    # Connect to existing ChromaDB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve top-k most relevant chunks (k=5 is a decent default)
    # Scores returned are *distances* (lower = closer/more relevant)
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine chunk text into one context block (separator helps the LLM)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Fill in the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Uncomment for debugging prompt issues
    # print(prompt)

    # Run local LLM via Ollama
    # Make sure you have it pulled already: ollama pull mistral
    model = OllamaLLM(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    # Pull chunk IDs so we can show where the answer came from
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    #print(formatted_response)

    return response_text


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    main()
