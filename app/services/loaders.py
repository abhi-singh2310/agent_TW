# app/services/loaders.py

"""
This module handles the loading and pre-processing of the source documentation.

In a RAG pipeline, it's crucial to prepare the source documents in a way that
maximises the effectiveness of the retrieval step. For this case study, the
source of truth is a PDF document. A key requirement is to cite the source
of the information, which includes the page number.

This loader implements a two-step process to ensure the page number metadata
is accurately preserved:
1.  It first converts the PDF into a structured text file, inserting clear
    markers (e.g., "--- PAGE X ---") at the beginning of each page's content.
    This is done only once and the result is cached as a `.txt` file.
2.  It then loads this structured text file and splits it into chunks.
    Because of the page markers, it can accurately assign the correct page
    number to the metadata of each chunk.

This approach is more reliable than directly loading the PDF with some
standard document loaders, which can sometimes struggle with complex layouts
and page number attribution.
"""

import logging
import re

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

from app import config

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _clean_extracted_text(text: str) -> str:
    """
    Cleans and normalises extracted PDF text to preserve Q&A structure.
    
    Args:
        text (str): Raw extracted PDF text.
        
    Returns:
        str: Cleaned text with proper Q&A formatting.
    """
    # Fix excessive spacing issues from PDF extraction
    text = re.sub(r'\s+', ' ', text)
    
    # Restore proper line breaks for Q&A structure
    text = re.sub(r'\s*Q:\s*', '\n\nQ: ', text)
    text = re.sub(r'\s*A:\s*', '\nA: ', text)
    
    # Fix bullet point formatting
    text = re.sub(r'\s*●\s*', '\n● ', text)
    
    # Clean up section headers
    text = re.sub(r'\s*(Assembly|Returns & Refunds|Shipping & Delivery|Product Care & Materials)\s*', '\n\n\1\n', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def _ensure_text_file_exists():
    """
    Checks if the structured text version of the PDF exists. If not, it
    creates it.

    This function is responsible for the one-time conversion of the source
    PDF into a text file with explicit page number markers. This avoids
    re-processing the PDF every time the application starts, which is more
    efficient.
    """
    text_file_path = config.DATA_DIR / "faq_manual.txt"
    pdf_file_path = config.PDF_FILE_PATH

    if text_file_path.exists():
        logger.info(
            "Structured text file already exists, skipping PDF conversion.")
        return

    logger.info(
        "Structured text file not found. Converting PDF to text with page "
        "numbers...")
    try:
        with open(pdf_file_path, "rb") as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            full_text = []
            # Extract text from each page and apply cleaning
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Clean the extracted text to preserve Q&A structure
                    cleaned_text = _clean_extracted_text(page_text)
                    full_text.append(f"--- PAGE {i + 1} ---\n{cleaned_text}")

        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write("\n\n".join(full_text))
        logger.info("Successfully converted PDF to clean structured text: %s",
                    text_file_path)

    except FileNotFoundError:
        logger.error("Error: The source PDF file was not found at %s",
                     pdf_file_path)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred during PDF conversion: %s",
                     e)
        raise


def load_and_split_pdf() -> List[Document]:
    """
    Loads the structured text file, splits it into chunks, and preserves the
    page number for each chunk.

    This is the main function for data loading. It ensures the structured
    text file exists, then processes it to create a list of `Document`
    objects, which are then split into smaller chunks suitable for a RAG
    pipeline.

    Returns:
        List[Document]: A list of chunked documents, each with correct
                        'source' and 'page' metadata.
    """
    try:
        # First, ensure the pre-processed text file is available.
        _ensure_text_file_exists()

        text_file_path = config.DATA_DIR / "faq_manual.txt"
        logger.info("Loading content from pre-processed text file: %s",
                    text_file_path)

        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # We split the text based on our page markers. This allows us to
        # process the content of each page individually.
        pages_content = re.split(r'--- PAGE \d+ ---\n', text)

        pages = []
        for i, content in enumerate(pages_content):
            if content.strip():  # Ensure content is not just whitespace.
                # The page number corresponds to the split index. This
                # correctly assigns the original PDF page number.
                pages.append(
                    Document(page_content=content,
                             metadata={
                                 "source": "faq_manual.pdf",
                                 "page": i
                             }))

        logger.info("Successfully loaded content from %d pages.", len(pages))

        # Now, we split the documents into smaller chunks. This is crucial
        # for RAG retrieval. We use Q&A-aware separators to preserve
        # complete question-answer pairs in FAQ documents.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            length_function=len,
            # Split at questions first to preserve Q&A structure
            separators=["\n\nQ:", "\n\n", "\n", ". ", " ", ""]
        )
        logger.info("Splitting document into Q&A-aware chunks...")
        chunks = text_splitter.split_documents(pages)
        logger.info(
            "Document split into %d chunks, each with correct page number "
            "metadata.", len(chunks))

        return chunks

    except Exception as e:
        logger.error("Failed to load and split content: %s", e)
        return []
