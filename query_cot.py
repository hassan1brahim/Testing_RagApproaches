# =============================================================================
# query_cot.py — RAG query engine with Chain of Thought prompting
# =============================================================================
#
# This is a variant of query.py that adds Chain of Thought reasoning.
# The LLM is prompted to reason step-by-step before answering.
#
# Differences from baseline (query.py):
# - Updated prompt template with reasoning steps
# - Everything else remains the same
#
# =============================================================================

import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from embed import get_embedding_function


# =============================================================================
# CONFIG
# =============================================================================

CHROMA_PATH = "chroma"


# =============================================================================
# PROMPT TEMPLATE — Chain of Thought
# =============================================================================
# Key difference: We ask the LLM to reason through the problem before answering.
# This often improves answer quality by making the reasoning explicit.
# =============================================================================

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}

Before answering, think through:
1. What information in the context is relevant to this question?
2. What reasoning steps connect the context to the answer?
3. Are there any gaps or uncertainties?

Reasoning:
<your step-by-step reasoning here>

Answer:
<your final answer here>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    response_text = query_rag(args.query_text)
    print(response_text)


def query_rag(query_text: str):
    """
    Runs RAG query with Chain of Thought prompting.
    Returns the full response (including reasoning).
    """

    embedding_function = get_embedding_function()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    return response_text


if __name__ == "__main__":
    main()
