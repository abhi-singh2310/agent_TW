# evaluate.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    chunks = load_and_split_pdf()
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
        "I bought a custom-made sofa but it arrived with a tear. Can I return it, and are there any shipping costs?",
        "What tools might be required for assembly, and what should I do if a part is missing?",
        "How do I care for my fabric sofa, and how often should I do it?",
        "Can I choose my delivery date, and what happens if I'm not home when the delivery arrives?",
        "A customer wants to know if they can get a replacement cushion cover for a sofa they bought. How should I handle this?"
    ]

    # --- Run Evaluation ---
    markdown_output = "# GenAI Agent Evaluation Results\n\n"
    for i, query in enumerate(questions):
        print(f"\n--- Testing Query {i+1}: {query} ---")
        
        start_time = time.time()
        response = agent.ask(query)
        end_time = time.time()
        
        latency = end_time - start_time
        
        print(f"Answer: {response['answer']}")
        print(f"Latency: {latency:.2f} seconds")

        # Format sources for Markdown
        source_list = []
        if response['sources']:
            for source in response['sources']:
                source_list.append(f"- `{source['source']}` | Location: Page {source['page']}")
        
        # Add to the Markdown output
        markdown_output += f"## Test Case {i+1}\n\n"
        markdown_output += f"**Question:** {query}\n"
        markdown_output += f"**Answer:** {response['answer']}\n"
        if source_list:
            markdown_output += f"**Sources:**\n"
            markdown_output += "\n".join(source_list) + "\n\n"
        
    # --- Save Results to Markdown file ---
    with open("evaluation_results.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)

    print("\n--- Evaluation Complete. Results saved to evaluation_results.md ---")

if __name__ == "__main__":
    main()