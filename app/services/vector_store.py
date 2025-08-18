# app/services/vector_store.py

import os
import logging
from typing import List, Sequence

# LangChain and community imports
from langchain.schema import Document, BaseRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings # Use FastEmbed
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

# Other libraries
from sentence_transformers.cross_encoder import CrossEncoder


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_embedding_model(model_name: str) -> FastEmbedEmbeddings:
    """
    Initializes and returns the FastEmbed embedding model.
    This is faster than using Ollama for embeddings.

    Args:
        model_name (str): The name of the model to use (e.g., 'BAAI/bge-small-en-v1.5').

    Returns:
        FastEmbedEmbeddings: An instance of the embedding model.
    """
    logger.info(f"Initializing FastEmbed embedding model: {model_name}")
    return FastEmbedEmbeddings(model_name=model_name)


def get_vector_store(
    chunks: List[Document],
    embedding_model: FastEmbedEmbeddings, # Updated type hint
    persist_directory: str
) -> Chroma:
    """
    Loads a ChromaDB vector store from disk if it exists, otherwise creates it.

    Args:
        chunks (List[Document]): A list of document chunks to add to the store
                                 if it needs to be created.
        embedding_model (FastEmbedEmbeddings): The embedding model to use.
        persist_directory (str): The directory to save the store to and load from.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    if os.path.exists(persist_directory):
        # If the directory exists, load the existing vector store
        logger.info(f"Loading existing vector store from: {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        logger.info("Vector store loaded successfully.")
    else:
        # If the directory does not exist, create a new vector store
        if not chunks:
            raise ValueError("Document chunks are required to create a new vector store.")

        logger.info(f"Creating new vector store and persisting to: {persist_directory}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        logger.info("New vector store created and persisted successfully.")

    return vector_store


def create_retriever(vector_store: Chroma, chunks: List[Document], top_k: int = 4) -> BaseRetriever:
    """
    Creates an ensemble retriever combining semantic and keyword search.

    Args:
        vector_store (Chroma): The vector store for semantic search.
        chunks (List[Document]): The list of document chunks for the BM25 retriever.
        top_k (int): The number of top relevant documents for each retriever to return.

    Returns:
        BaseRetriever: A LangChain EnsembleRetriever object.
    """
    logger.info(f"Creating retriever to fetch top {top_k} documents.")

    # 1. Create the semantic retriever (from the vector store)
    semantic_retriever = vector_store.as_retriever(search_kwargs={'k': top_k})

    # 2. Create the keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    # 3. Create the ensemble retriever to combine both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]  # Adjust weights to balance between semantic and keyword search
    )

    return ensemble_retriever


class LocalRerankCompressor(BaseDocumentCompressor):
    """
    A document compressor that uses a local cross-encoder model to rerank documents.
    This improves the final relevance of documents before sending them to the LLM.
    """
    model: CrossEncoder
    top_n: int

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks = None,
    ) -> Sequence[Document]:
        """
        Reranks the documents based on the query.
        """
        if not documents:
            return []

        # Create pairs of [query, document_content] for the cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]

        # Get the scores from the cross-encoder model
        scores = self.model.predict(pairs)

        # Combine documents and scores, then sort
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return the top_n documents
        return [doc for _, doc in scored_docs[:self.top_n]]


def create_reranker_retriever(
    ensemble_retriever: BaseRetriever,
    model_name: str,
    top_n: int = 3
) -> BaseRetriever:
    """
    Creates a reranker retriever using a local cross-encoder model.

    Args:
        ensemble_retriever (BaseRetriever): The retriever to get initial results from.
        model_name (str): The name of the cross-encoder model to use.
        top_n (int): The number of top documents to return after reranking.

    Returns:
        BaseRetriever: A reranking retriever.
    """
    logger.info(f"Creating local reranker with model: {model_name}")

    # Initialize the cross-encoder model
    cross_encoder = CrossEncoder(model_name)

    # Create the local rerank compressor
    compressor = LocalRerankCompressor(model=cross_encoder, top_n=top_n)

    # Create the contextual compression retriever which wraps the base retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return compression_retriever