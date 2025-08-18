# app/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Project Root ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data and Database Paths ---
DATA_DIR = BASE_DIR / "data"
PDF_FILE_PATH = DATA_DIR / "faq_manual.pdf"
VECTOR_STORE_DIR = BASE_DIR / "db"
VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "chroma_db")

# --- AI Model Configuration ---
# Name of the local model to use for the embedding process.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Hugging Face LLM Configuration ---
LLM_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# Get the API key from the environment variables you just set up.
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Retriever Configuration ---
RETRIEVER_TOP_K = 15

# --- Reranker Configuration ---
RERANKER_TOP_N = 4
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"