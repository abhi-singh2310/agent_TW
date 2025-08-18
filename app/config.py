# app/config.py

from pathlib import Path

# --- Project Root ---
# This reliably finds the project's root directory, making paths portable.
# Path(__file__) is this file -> .parent is the 'app' directory -> .parent is the project root.
BASE_DIR = Path(__file__).resolve().parent.parent


# --- Data and Database Paths ---
# Path to the directory where source documents are stored.
DATA_DIR = BASE_DIR / "data"

# Full path to the source PDF file.
PDF_FILE_PATH = DATA_DIR / "faq_manual.pdf"

# Path to the directory where the ChromaDB vector store will be persisted.
VECTOR_STORE_DIR = BASE_DIR / "db"
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "chroma_db") # ChromaDB needs the path as a string


# --- AI Model Configuration ---
# Name of the local model to use for the embedding process.
EMBEDDING_MODEL_NAME = "nomic-embed-text"

# Name of the local model to use for the generation (chat) process.
LLM_MODEL_NAME = "phi3:3.8b-mini-128k-instruct-q4_0"


# --- Retriever Configuration ---
# The number of top relevant document chunks to retrieve for a given query.
RETRIEVER_TOP_K = 15

# --- Reranker Configuration ---
# The number of top documents to return after reranking.
RERANKER_TOP_N = 3
# The name of the local cross-encoder model to use for reranking.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
