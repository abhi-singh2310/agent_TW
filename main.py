# main.py

"""
This module serves as the entry point for the GenAI-powered FAQ agent.

It initialises and runs a Flask web server that provides a simple web
interface for interacting with the agent. The server exposes a single API
endpoint (`/ask`) that accepts user queries and returns the agent's
generated answer along with the sources it used.

The initialisation of the agent and all its components (data loading,
embedding model, vector store, and retriever) is handled here.
"""

import logging

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from app import config
from app.agent import GenAIAgent
from app.services.loaders import load_and_split_pdf
from app.services.vector_store import (create_embedding_model,
                                        create_reranker_retriever,
                                        create_retriever, get_vector_store)

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Agent Initialisation ---
# The agent is initialised as a global variable to ensure that the expensive
# setup process (loading models, creating the vector store) only happens once
# when the application starts.
agent = None

# This flag allows for easy toggling of the reranker. The reranker improves
# the quality of the retrieved documents but adds a small amount of latency.
# Having this as a configurable flag is useful for performance testing and
# demonstrations.
USE_RERANKER = True


def initialize_agent():
    """
    Initialises all components required for the GenAI agent.

    This function orchestrates the entire setup process for the agent. It's
    designed to be called once at startup. The steps are logged to the
    console to provide visibility into the initialisation process for debugging.

    The process involves:
    1.  Loading the source document (the FAQ PDF) and splitting it into
        manageable chunks.
    2.  Initialising the embedding model, which is used to convert the text
        chunks into numerical vectors.
    3.  Creating or loading the vector store, which stores the embeddings and
        allows for efficient similarity searches.
    4.  Creating a retriever, which is responsible for fetching the most
        relevant documents from the vector store based on a user's query.
    5.  Optionally, creating a reranker to further refine the retrieved
        documents.
    6.  Finally, initialising the GenAIAgent with the fully configured
        retriever and the specified language model.
    """
    global agent
    if agent is not None:
        logger.info("Agent is already initialised.")
        return

    logger.info("--- Starting Agent Initialisation ---")

    logger.info(">>> STEP 1: Loading and splitting PDF...")
    chunks = load_and_split_pdf()
    logger.info("<<< STEP 1: Complete.\n")

    logger.info(
        ">>> STEP 2: Initialising embedding model (might download model)...")
    embedding_model = create_embedding_model(
        model_name=config.EMBEDDING_MODEL_NAME)
    logger.info("<<< STEP 2: Complete.\n")

    logger.info(
        ">>> STEP 3: Getting vector store (might re-create embeddings)...")
    vector_store = get_vector_store(chunks=chunks,
                                    embedding_model=embedding_model,
                                    persist_directory=config.VECTOR_STORE_PATH)
    logger.info("<<< STEP 3: Complete.\n")

    logger.info(">>> STEP 4: Creating ensemble retriever...")
    ensemble_retriever = create_retriever(vector_store=vector_store,
                                          chunks=chunks,
                                          top_k=config.RETRIEVER_TOP_K)
    logger.info("<<< STEP 4: Complete.\n")

    final_retriever = ensemble_retriever  # Default if reranker is off

    if USE_RERANKER:
        logger.info(
            ">>> STEP 5: Creating reranker retriever (might download model)...")
        final_retriever = create_reranker_retriever(
            ensemble_retriever=ensemble_retriever,
            model_name=config.RERANKER_MODEL_NAME,
            top_n=config.RERANKER_TOP_N)
        logger.info("<<< STEP 5: Complete.\n")
    else:
        logger.info(">>> STEP 5: Skipping reranker.\n")

    logger.info(">>> STEP 6: Initialising GenAI Agent...")
    agent = GenAIAgent(retriever=final_retriever,
                       llm_model_name=config.LLM_MODEL_NAME)
    logger.info("<<< STEP 6: Complete.\n")
    logger.info("--- Agent Initialisation Complete ---")


# --- Flask Application Setup ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable Cross-Origin Resource Sharing for the frontend


# --- API Endpoints ---
@app.route('/')
def serve_index():
    """Serves the main HTML page for the web interface."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/ask', methods=['POST'])
def ask_agent():
    """
    Handles questions to the agent and returns a JSON response.

    This is the primary API endpoint for the application. It expects a POST
    request with a JSON body containing the user's 'query' and optional
    'history'. It performs basic validation and error handling.
    """
    if not agent:
        return jsonify({"error": "Agent not initialised"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')
    history = data.get('history')

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    try:
        response = agent.ask(query, history=history)
        return jsonify(response)
    except Exception as e:
        logger.error("An error occurred while processing the query: %s", e)
        return jsonify({"error": "An internal error occurred."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    initialize_agent()
    # Note: `debug=True` is for development.
    app.run(host='127.0.0.1', port=5000, debug=True)
