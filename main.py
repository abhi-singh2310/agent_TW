# main.py

import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import configurations and core components
from app import config
from app.agent import GenAIAgent
from app.services.loaders import load_and_split_pdf
from app.services.vector_store import (
    create_embedding_model,
    get_vector_store,
    create_retriever,
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
    # Chunks are only needed if the vector store doesn't exist yet.
    # We load them first to pass them to the get_vector_store function if needed.
    chunks = load_and_split_pdf(str(config.PDF_FILE_PATH))

    # 2. Initialize embedding model
    embedding_model = create_embedding_model(model_name=config.EMBEDDING_MODEL_NAME)

    # 3. Get or create the vector store
    vector_store = get_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
        persist_directory=config.VECTOR_STORE_PATH
    )

    # 4. Create the retriever
    retriever = create_retriever(
        vector_store=vector_store,
        top_k=config.RETRIEVER_TOP_K
    )

    # 5. Initialize the GenAI Agent
    agent = GenAIAgent(
        retriever=retriever,
        llm_model_name=config.LLM_MODEL_NAME
    )
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
    Handles questions to the agent. Expects a JSON payload with a 'query' key.
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