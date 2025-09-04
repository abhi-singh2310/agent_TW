# debug_chunks.py

"""
Simple debugging script to inspect chunk creation and Q&A structure.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from app.services.loaders import load_and_split_pdf

def debug_chunks():
    """Debug the chunking process to see Q&A structure preservation."""
    
    print("=== CHUNK DEBUGGING ===\n")
    
    # Load chunks using current logic
    chunks = load_and_split_pdf()
    
    print(f"Total chunks created: {len(chunks)}\n")
    
    # Look for specific Q&A pairs we're testing
    test_questions = [
        "How do I care for my wooden furniture",
        "What is the return policy", 
        "How do I track my order",
        "Is assembly assistance available"
    ]
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        page = chunk.metadata.get('page', 'unknown')
        
        # Check if this chunk contains any of our test Q&As
        for question in test_questions:
            if question.lower() in content.lower():
                print(f"=== CHUNK {i+1} (Page {page}) - CONTAINS: '{question}' ===")
                print(f"Length: {len(content)} characters")
                print(f"Content:\n{content}")
                print(f"{'='*60}\n")
                break
    
    # Also show a few random chunks for general inspection
    print("=== SAMPLE CHUNKS FOR GENERAL INSPECTION ===\n")
    for i in [0, len(chunks)//2, -1]:
        if i < len(chunks):
            chunk = chunks[i]
            print(f"Chunk {i+1} (Page {chunk.metadata.get('page')}):")
            print(f"Length: {len(chunk.page_content)} chars")
            print(f"Preview: {chunk.page_content[:200]}...")
            print(f"{'-'*40}\n")

if __name__ == "__main__":
    debug_chunks()
