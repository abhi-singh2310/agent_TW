# main.py

import logging
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

# Import configurations and core components
from app import config
from app.agent import GenAIAgent
from app.services.loaders import load_and_split_pdf
from app.services.vector_store import (
    create_embedding_model,
    get_vector_store,
    create_retriever,
    create_reranker_retriever,
)

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Agent Initialization ---
# This setup runs only once when the server starts.
agent = None

def initialize_agent():
    """
    Initializes all components required for the GenAI agent.
    This includes loading data, creating embeddings, the vector store,
    the retriever, and finally the agent itself.
    """
    global agent
    if agent is not None:
        logger.info("Agent is already initialized.")
        return

    logger.info("--- Starting Agent Initialization ---")

    # 1. Load and split the document
    logger.info("Step 1: Loading and splitting PDF...")
    chunks = load_and_split_pdf(str(config.PDF_FILE_PATH))
    logger.info("Step 1: Complete.")

    # 2. Initialize embedding model
    logger.info("Step 2: Initializing embedding model...")
    embedding_model = create_embedding_model(model_name=config.EMBEDDING_MODEL_NAME)
    logger.info("Step 2: Complete.")

    # 3. Get or create the vector store
    logger.info("Step 3: Getting vector store...")
    vector_store = get_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
        persist_directory=config.VECTOR_STORE_PATH
    )
    logger.info("Step 3: Complete.")

    # 4. Create the ensemble retriever
    logger.info("Step 4: Creating ensemble retriever...")
    ensemble_retriever = create_retriever(
        vector_store=vector_store,
        chunks=chunks,
        top_k=config.RETRIEVER_TOP_K
    )
    logger.info("Step 4: Complete.")

    # 5. Create the reranker retriever
    logger.info("Step 5: Creating reranker retriever... (This may take a moment)")
    reranker_retriever = create_reranker_retriever(
        ensemble_retriever=ensemble_retriever,
        model_name=config.RERANKER_MODEL_NAME,
        top_n=config.RERANKER_TOP_N
    )
    logger.info("Step 5: Complete.")

    # 6. Initialize the GenAI Agent
    logger.info("Step 6: Initializing GenAI Agent...")
    agent = GenAIAgent(
        retriever=reranker_retriever,
        llm_model_name=config.LLM_MODEL_NAME
    )
    logger.info("Step 6: Complete.")
    logger.info("--- Agent Initialization Complete ---")


# --- Flask Application Setup ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app) # Enable Cross-Origin Resource Sharing for the frontend

# --- API Endpoints ---
@app.route('/')
def serve_index():
    """Serves the main HTML page for the demo."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/ask', methods=['POST'])
def ask_agent():
    """
    Handles questions to the agent and returns a single JSON response.
    """
    if not agent:
        return jsonify({"error": "Agent not initialized"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    try:
        # Get the complete response from the agent
        response = agent.ask(query)
        return jsonify(response)
    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}")
        return jsonify({"error": "An internal error occurred."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    initialize_agent()
    # Note: debug=True is for development. In production, use a proper WSGI server like Gunicorn.
    app.run(host='127.0.0.1', port=5000, debug=True)
