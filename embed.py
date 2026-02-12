# =============================================================================
# embed.py — embedding function used for both indexing + querying
# =============================================================================
#
# This file defines the single embedding model used across the project.
# Embeddings are what enable semantic (meaning-based) search.
#
# Important rule:
# - The SAME embedding model must be used for indexing and querying.
#   Mixing models will break retrieval because the vectors won’t align.
#
# Current status:
# - I’m using Ollama embeddings right now since I don’t have access to
#   AWS Bedrock at the moment.
# - In past experiments, Bedrock (Titan) embeddings have generally
#   produced better retrieval quality.
# =============================================================================

# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    """
    Returns the embedding function used everywhere in the RAG pipeline.

    Used in:
    - load.py (embedding document chunks)
    - query.py (embedding user queries)

    Note:
    Bedrock embeddings have worked better overall, but Ollama is being
    used for now since it runs locally and doesn’t require AWS access.
    """

    # Local embeddings via Ollama (current setup)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Preferred setup (when Bedrock access is available again):
    #
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default",
    #     region_name="us-east-1"
    # )
    #
    # Better embeddings → better retrieval → better RAG answers.

    return embeddings

# Wrapped in a function to keep the embedding choice centralized
# and easy to swap later.
