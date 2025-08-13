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
    embedding_model: OllamaEmbeddings, 
    persist_directory: str
) -> Chroma:
    """
    Creates a new ChromaDB vector store or loads an existing one.

    If the `persist_directory` already exists, it loads the database from there.
    Otherwise, it creates a new database from the provided document chunks,
    computes the embeddings, and saves it to the directory for future use.

    Args:
        chunks (List[Document]): The list of document chunks to be embedded.
        embedding_model (OllamaEmbeddings): The model used to create embeddings.
        persist_directory (str): The directory to save/load the vector store.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    if os.path.exists(persist_directory):
        logger.info(f"Loading existing vector store from: {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    else:
        if not chunks:
            raise ValueError("Document chunks are required to create a new vector store.")
        logger.info(f"Creating new vector store at: {persist_directory}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        logger.info("Vector store created and persisted successfully.")
        
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