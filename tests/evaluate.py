# evaluate.py

import time
import pandas as pd
from app.agent import GenAIAgent
from app.services.loaders import load_and_split_pdf
from app.services.vector_store import (
    create_embedding_model,
    get_vector_store,
    create_retriever,
)
import app.config as config

def main():
    # --- Initialize the Agent (similar to main.py) ---
    print("--- Initializing Agent ---")
    chunks = load_and_split_pdf(str(config.PDF_FILE_PATH))
    embedding_model = create_embedding_model(model_name=config.EMBEDDING_MODEL_NAME)
    vector_store = get_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
        persist_directory=config.VECTOR_STORE_PATH
    )
    retriever = create_retriever(
        vector_store=vector_store,
        chunks=chunks,
        top_k=config.RETRIEVER_TOP_K
    )
    agent = GenAIAgent(
        retriever=retriever,
        llm_model_name=config.LLM_MODEL_NAME
    )
    print("--- Agent Initialized ---")

    # --- Define Evaluation Questions ---
    questions = [
        "What is the return policy?",
        "How can I track my order?",
        "What are the shipping options?",
        # Add more questions relevant to your PDF
    ]

    # --- Run Evaluation ---
    results = []
    for query in questions:
        print(f"\n--- Testing Query: {query} ---")
        
        start_time = time.time()
        response = agent.ask(query)
        end_time = time.time()
        
        latency = end_time - start_time
        
        results.append({
            "question": query,
            "answer": response["answer"],
            "sources": response["sources"],
            "latency_seconds": latency,
        })
        
        print(f"Answer: {response['answer']}")
        print(f"Latency: {latency:.2f} seconds")

    # --- Save Results to CSV ---
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("\n--- Evaluation Complete. Results saved to evaluation_results.csv ---")

if __name__ == "__main__":
    main()