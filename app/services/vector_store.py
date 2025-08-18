# app/services/vector_store.py

"""
This module is responsible for creating and managing the vector store and the
retrieval components of the RAG pipeline.

The key components are:
1.  **Embedding Model:** Uses `FastEmbed` for efficient, local creation of
    text embeddings.
2.  **Vector Store:** Uses `ChromaDB` to store the document embeddings and
    perform semantic similarity searches. The vector store is persisted to
    disk to avoid re-creating it on every application start.
3.  **Ensemble Retriever:** Combines the strengths of two different retrieval
    methods:
    -   **Semantic Search:** Finds documents that are conceptually related to
        the user's query.
    -   **Keyword Search (BM25):** Finds documents that contain the exact
        keywords from the query.
    This hybrid approach, known as hybrid search, is often more robust than
    relying on a single retrieval method.
4.  **Reranker:** A `CrossEncoder` model is used to rerank the documents
    retrieved by the ensemble retriever. This adds a final layer of quality
    control, ensuring that only the most relevant documents are passed to the
    language model, which is crucial for generating accurate answers.
"""

import logging
import os
from typing import List, Sequence

from langchain.retrievers import (ContextualCompressionRetriever,
                                  EnsembleRetriever)
from langchain.retrievers.document_compressors.base import \
    BaseDocumentCompressor
from langchain.schema import BaseRetriever, Document
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers.cross_encoder import CrossEncoder

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_embedding_model(model_name: str) -> FastEmbedEmbeddings:
    """
    Initialises and returns the FastEmbed embedding model.

    `FastEmbed` is chosen here for its speed and efficiency. It runs locally
    and is optimised for creating embeddings quickly, which is ideal for the
    initial setup phase of the agent.

    Args:
        model_name (str): The name of the model to use (e.g.,
                          'BAAI/bge-small-en-v1.5').

    Returns:
        FastEmbedEmbeddings: An instance of the embedding model.
    """
    logger.info("Initialising FastEmbed embedding model: %s", model_name)
    return FastEmbedEmbeddings(model_name=model_name)


def get_vector_store(chunks: List[Document],
                     embedding_model: FastEmbedEmbeddings,
                     persist_directory: str) -> Chroma:
    """
    Loads a ChromaDB vector store from disk if it exists, otherwise creates it.

    Persisting the vector store is a critical optimisation. The process of
    creating embeddings for all the document chunks can be time-consuming.
    By saving the store to disk, we ensure that this process only needs to be
    done once.

    Args:
        chunks (List[Document]): A list of document chunks to add to the store
                                 if it needs to be created.
        embedding_model (FastEmbedEmbeddings): The embedding model to use.
        persist_directory (str): The directory to save the store to and load
                                 from.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    if os.path.exists(persist_directory):
        logger.info("Loading existing vector store from: %s",
                    persist_directory)
        vector_store = Chroma(persist_directory=persist_directory,
                              embedding_function=embedding_model)
        logger.info("Vector store loaded successfully.")
    else:
        if not chunks:
            raise ValueError(
                "Document chunks are required to create a new vector store.")

        logger.info("Creating new vector store and persisting to: %s",
                    persist_directory)
        vector_store = Chroma.from_documents(documents=chunks,
                                             embedding=embedding_model,
                                             persist_directory=persist_directory)
        logger.info("New vector store created and persisted successfully.")

    return vector_store


def create_retriever(vector_store: Chroma, chunks: List[Document],
                     top_k: int = 4) -> BaseRetriever:
    """
    Creates an ensemble retriever combining semantic and keyword search.

    This function builds a hybrid search retriever. It combines a semantic
    retriever (from the Chroma vector store) with a traditional keyword
    retriever (BM25). This approach is powerful because it can handle a wide
    variety of queries. Semantic search is great for understanding the
    *meaning* behind a query, while keyword search is excellent for finding
    documents with specific terms or product names.

    Args:
        vector_store (Chroma): The vector store for semantic search.
        chunks (List[Document]): The list of document chunks for the BM25
                                 retriever.
        top_k (int): The number of top relevant documents for each retriever
                     to return.

    Returns:
        BaseRetriever: A LangChain EnsembleRetriever object.
    """
    logger.info("Creating retriever to fetch top %d documents.", top_k)

    # 1. Create the semantic retriever.
    semantic_retriever = vector_store.as_retriever(search_kwargs={'k': top_k})

    # 2. Create the keyword retriever.
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    # 3. Create the ensemble retriever. The weights determine the relative
    #    importance of each retriever's results. Here, we give a slight
    #    preference to semantic search.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4])

    return ensemble_retriever


class LocalRerankCompressor(BaseDocumentCompressor):
    """
    A document compressor that uses a local cross-encoder model to rerank
    documents.

    A reranker provides a more sophisticated scoring of document relevance
    than the initial retrieval step. While the retriever quickly finds a set
    of potentially relevant documents, the reranker then examines this smaller 
    set in more detail to provide a final, more accurate ranking.
    """
    model: CrossEncoder
    top_n: int

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """Reranks the documents based on the query."""
        if not documents:
            return []

        # The cross-encoder model expects pairs of [query, document_content].
        pairs = [[query, doc.page_content] for doc in documents]

        # The model predicts a relevance score for each pair.
        scores = self.model.predict(pairs)

        # We combine the documents and scores, then sort to get the most
        # relevant documents at the top.
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Finally, we return the top_n documents.
        return [doc for _, doc in scored_docs[:self.top_n]]


def create_reranker_retriever(ensemble_retriever: BaseRetriever,
                              model_name: str,
                              top_n: int = 3) -> BaseRetriever:
    """
    Creates a reranker retriever using a local cross-encoder model.

    This function wraps the base `ensemble_retriever` with a
    `ContextualCompressionRetriever`. The "compressor" in this case is our
    `LocalRerankCompressor`, which doesn't actually compress the documents,
    but rather reranks them and selects the top N.

    Args:
        ensemble_retriever (BaseRetriever): The retriever to get initial
                                            results from.
        model_name (str): The name of the cross-encoder model to use.
        top_n (int): The number of top documents to return after reranking.

    Returns:
        BaseRetriever: A reranking retriever.
    """
    logger.info("Creating local reranker with model: %s", model_name)

    cross_encoder = CrossEncoder(model_name)
    compressor = LocalRerankCompressor(model=cross_encoder, top_n=top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever)

    return compression_retriever
