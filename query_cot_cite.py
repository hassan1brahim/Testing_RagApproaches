# =============================================================================
# query_cot_cite.py — RAG query engine with Chain of Thought + Citations
# =============================================================================
#
# This is a variant of query.py that adds:
# 1. Chain of Thought reasoning (like query_cot.py)
# 2. Source citation tracking
#
# Differences from baseline (query.py):
# - Context chunks are labeled with [Source N] and metadata
# - Prompt instructs LLM to cite sources in the answer
# - Returns a dict with answer + structured source information
#
# =============================================================================

import argparse
import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from embed import get_embedding_function


# =============================================================================
# CONFIG
# =============================================================================

CHROMA_PATH = "chroma"


# =============================================================================
# PROMPT TEMPLATE — Chain of Thought + Citations
# =============================================================================
# Key differences:
# - Context pieces are labeled with source numbers
# - LLM is instructed to cite sources using [Source N] notation
# =============================================================================

PROMPT_TEMPLATE = """
Answer the question based only on the following context. Each piece of context is labeled with a source number.

{context}

---

Question: {question}

Before answering, think through:
1. What information in the context is relevant to this question?
2. What reasoning steps connect the context to the answer?
3. Are there any gaps or uncertainties?

When you use information from the context, cite the source using [Source N] notation.

Reasoning:
<your step-by-step reasoning here>

Answer:
<your final answer with [Source N] citations>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    result = query_rag(args.query_text)

    print(f"\nQuestion: {result['question']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for src in result['sources']:
        print(f"  - Source {src['source_num']}: {src['filename']}, Page {src['page']}, Chunk: {src['chunk_id']}")


def query_rag(query_text: str) -> dict:
    """
    Runs RAG query with Chain of Thought prompting and citation tracking.

    Returns:
        dict with keys:
        - question: the original query
        - answer: the LLM response (with reasoning + citations)
        - sources: list of source metadata dicts
    """

    embedding_function = get_embedding_function()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    # Build context with source labels
    context_parts = []
    sources = []

    for i, (doc, score) in enumerate(results, 1):
        source_path = doc.metadata.get("source", "Unknown")
        filename = os.path.basename(source_path)
        page = doc.metadata.get("page", 0)
        chunk_id = doc.metadata.get("id", "Unknown")

        # Label each context chunk
        context_parts.append(
            f"[Source {i}] (from {filename}, Page {page + 1}):\n{doc.page_content}"
        )

        # Track source metadata
        sources.append({
            "source_num": i,
            "filename": filename,
            "page": page + 1,  # 1-indexed for display
            "chunk_id": chunk_id,
            "score": score
        })

    context_text = "\n\n---\n\n".join(context_parts)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    return {
        "question": query_text,
        "answer": response_text,
        "sources": sources
    }


if __name__ == "__main__":
    main()
