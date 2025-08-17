# app/services/vector_store.py

import os
import logging
from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, BaseRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_embedding_model(model_name: str) -> OllamaEmbeddings:
    """
    Initializes and returns the Ollama embedding model.

    Args:
        model_name (str): The name of the model to use via Ollama (e.g., 'mistral').

    Returns:
        OllamaEmbeddings: An instance of the embedding model.
    """
    logger.info(f"Initializing embedding model: {model_name}")
    return OllamaEmbeddings(model=model_name)

def get_vector_store(
    chunks: List[Document],
    embedding_model: OllamaEmbeddings
) -> Chroma:
    """
    Creates a new in-memory ChromaDB vector store.
    """
    if not chunks:
        raise ValueError("Document chunks are required to create the vector store.")

    logger.info("Creating new in-memory vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    logger.info("In-memory vector store created successfully.")
    return vector_store

def create_retriever(vector_store: Chroma, top_k: int = 4) -> BaseRetriever:
    """
    Creates a retriever from a vector store.

    The retriever is configured to find the 'top_k' most relevant documents
    for a given query.

    Args:
        vector_store (Chroma): The vector store to retrieve documents from.
        top_k (int): The number of top relevant documents to return.

    Returns:
        BaseRetriever: A LangChain retriever object.
    """
    logger.info(f"Creating retriever to fetch top {top_k} documents.")
    return vector_store.as_retriever(search_kwargs={'k': top_k})