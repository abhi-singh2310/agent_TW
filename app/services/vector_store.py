# app/services/vector_store.py

import os
import logging
from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers.cross_encoder import CrossEncoder
from typing import Sequence

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
    Loads a ChromaDB vector store from disk if it exists, otherwise creates it.

    Args:
        chunks (List[Document]): A list of document chunks to add to the store
                                 if it needs to be created.
        embedding_model (OllamaEmbeddings): The embedding model to use.
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
    Creates a retriever from a vector store.

    The retriever is configured to find the 'top_k' most relevant documents
    for a given query.

    Args:
        vector_store (Chroma): The vector store to retrieve documents from.
        chunks (List[Document]): The list of document chunks for the BM25 retriever.
        top_k (int): The number of top relevant documents to return.

    Returns:
        BaseRetriever: A LangChain retriever object.
    """
    logger.info(f"Creating retriever to fetch top {top_k} documents.")
    
    # 1. Create the semantic retriever
    semantic_retriever = vector_store.as_retriever(search_kwargs={'k': top_k})
    
    # 2. Create the keyword retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k
    
    # 3. Create the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # You can tune these weights
    )
    
    return ensemble_retriever

class LocalRerankCompressor(BaseDocumentCompressor):
    """
    A document compressor that uses a local cross-encoder model to rerank documents.
    """
    model: CrossEncoder
    top_n: int

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks = None,
    ) -> Sequence[Document]:
        """
        Reranks the documents based on the query.

        Args:
            documents (Sequence[Document]): The documents to rerank.
            query (str): The query to use for reranking.
            callbacks: Optional callbacks.

        Returns:
            Sequence[Document]: The top_n reranked documents.
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
    
    # Create the contextual compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever
