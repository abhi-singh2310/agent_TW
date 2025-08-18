# GenAI-Powered FAQ & Document Search Agent

## Project Overview

This project contains a GenAI-powered agent designed to assist customer support teams by automating responses to repetitive customer queries.

The agent ingests a knowledge base (in this case, a PDF containing FAQs), and provides a simple web interface for users to ask questions in natural language. The agent then generates a concise, context-aware answer and, crucially, cites the sources from the original document, including the page number.

This solution is built using Python (back-end), simple JS and CSS (front-end) and a stack of open-source tools, primarily the LangChain framework. GenAI (Gemini 2.5 Pro) was used to workshop, build, test and debug. This was an intentional choice to build something quickly and efficiently.

## Approach

The agent is built using a **Retrieval-Augmented Generation (RAG)** architecture. This approach was chosen because it is highly effective for building question-answering systems over a specific body of knowledge. The key advantage of RAG is that it grounds the language model's responses in the provided documentation, which significantly reduces the risk of the model "hallucinating" or making up incorrect information.

The RAG pipeline is implemented as follows:

1.  **Document Loading and Processing:** The source PDF is pre-processed into a structured text file with page number markers. This text is then split into small, overlapping chunks. This ensures that the page number for each piece of information is accurately preserved.

2.  **Vector Store and Embeddings:** The document chunks are converted into numerical vectors (embeddings) using a `FastEmbed` model. These embeddings are stored in a `ChromaDB` vector store, which allows for efficient semantic similarity searches. The vector store is persisted to disk to avoid re-processing the documents on every application start.

3.  **Hybrid Retrieval:** To find the most relevant information for a given query, a hybrid retrieval strategy is used. This combines:
    *   **Semantic Search:** To find documents that are conceptually related to the query.
    *   **Keyword Search (BM25):** To find documents that contain the exact keywords from the query.
    This hybrid approach is more robust than relying on a single retrieval method.

4.  **Reranking:** The documents retrieved by the hybrid retriever are then passed to a `CrossEncoder` model for reranking. This adds a final layer of quality control, ensuring that only the most relevant documents are passed to the language model.

5.  **Answer Generation:** The top-ranked documents are then formatted into a prompt, along with the user's query and the conversation history. This prompt is then passed to a large language model (LLM) from the Hugging Face Hub, which generates the final, context-aware answer.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── agent.py              # Core GenAIAgent class and RAG chain logic.
│   ├── config.py             # Configuration settings (e.g., model names, file paths).
│   └── services/
│       ├── __init__.py
│       ├── loaders.py        # Logic for loading and processing the source PDF.
│       └── vector_store.py   # Logic for creating the vector store and retrievers.
├── data/
│   └── faq_manual.pdf        # The source knowledge base.
├── db/
│   └── chroma_db/            # Persisted ChromaDB vector store.
├── static/
│   ├── css/
│   ├── images/
│   ├── js/
│   └── index.html            # The simple web interface for the agent.
├── tests/
│   └── ...                   # Test cases for the agent.
├── .env                      # Environment variables (e.g., API keys).
├── main.py                   # Flask web server and application entry point.
├── requirements.txt          # Project dependencies.
├── venv/                     # Python virtual environment.
└── README.md                 # This file.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abhi-singh2310/agent_TW.git
    cd agent_TW
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies (upgrade pip to ensure completion):**
    ```bash
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Set up your Hugging Face API key:**
    This project requires a Hugging Face API key to access the language model. You can get one for free from the [Hugging Face website](https://huggingface.co/settings/tokens). Create an account and get a token.

    Create a `.env` file in the root of the project and add your API key:
    ```
    HUGGINGFACE_API_KEY="your_api_key_here"
    ```

## How to Run

1.  **Start the Flask application:**
    The initial setup may take some time to load but later runs should be far smoother.

    ```bash
    python main.py
    ```
    This will start the web server on `http://127.0.0.1:5000`.

2.  **Open the web interface:**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

3.  **Ask a question:**
    Type or choose a question in the chat interface and press Enter. The agent will generate an answer and provide the sources it used.

## Limitations

*   **Knowledge Cut-off:** The agent's knowledge is limited to the content of the `faq_manual.pdf` file. It cannot answer questions about topics not covered in this document.
*   **Scalability:** The current implementation uses a local vector store and runs on a single machine. For a production environment with a large number of users, it would be necessary to use a more scalable, cloud-based vector store and deploy the application on a more robust infrastructure.
*   **Model Dependencies:** The performance of the agent is dependent on the quality of the chosen embedding, reranker, and language models. The current models were chosen as good open-source options, but other models may provide better performance.
*   **Local vs Cloud LLM Consideration:** For this task, I initially considered the use of a locally-run LLM using Ollama such as mistral:7b or phi3:mini but ran into significant performance issues due to hardware limitations of my Macbook Pro. For this reason, I pivoted to a cloud instance with the well-known huggingface suite of models that enabled fast and effective calls to be made.
*   **Accuracy vs Speed:** It was a battle to identify where the parameters could be configured to facilitate fast but accurate calls to the LLM. Changes were tested between different chunking sizes, output tokens, retrieval and encoding strategies as well as persistent memory options. The current selection was identified to be working from a trial and error approach.
