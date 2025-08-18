# app/services/loaders.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging
import pypdf
import re # Import the regular expressions library
from app import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _ensure_text_file_exists():
    """
    Checks if the structured text version of the PDF exists. If not, it creates it.
    This new version includes page number markers.
    """
    text_file_path = config.DATA_DIR / "faq_manual.txt"
    pdf_file_path = config.PDF_FILE_PATH

    if text_file_path.exists():
        logger.info(f"Structured text file already exists, skipping PDF conversion.")
        return

    logger.info(f"Structured text file not found. Converting PDF to text with page numbers...")
    try:
        with open(pdf_file_path, "rb") as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            full_text = []
            # --- CHANGE 1: Process PDF page by page ---
            # Loop through each page to keep track of the page number.
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Add a clear marker for each page's content.
                    full_text.append(f"--- PAGE {i + 1} ---\n{page_text}")

        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write("\n\n".join(full_text))
        logger.info(f"Successfully converted PDF to structured text: {text_file_path}")

    except FileNotFoundError:
        logger.error(f"Error: The source PDF file was not found at {pdf_file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during PDF conversion: {e}")
        raise


def load_and_split_pdf() -> List[Document]:
    """
    Ensures a structured text version of the PDF exists, then loads and splits it,
    preserving the correct page number for each chunk.
    """
    try:
        # Step 1: Make sure the structured .txt file is available
        _ensure_text_file_exists()

        text_file_path = config.DATA_DIR / "faq_manual.txt"
        logger.info(f"Loading content from pre-processed text file: {text_file_path}")

        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # --- CHANGE 2: Load from the structured text file ---
        # Split the text by our "--- PAGE X ---" marker to process each page's content.
        pages_content = re.split(r'--- PAGE \d+ ---\n', text)
        
        pages = []
        for i, content in enumerate(pages_content):
            if content.strip(): # Ensure content is not just whitespace
                # The page number corresponds to the split index (plus 1).
                # This correctly assigns the original PDF page number as metadata.
                pages.append(Document(page_content=content, metadata={"source": "faq_manual.pdf", "page": i}))
        
        logger.info(f"Successfully loaded content from {len(pages)} pages.")

        # Step 3: Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=250,
            length_function=len
        )
        logger.info("Splitting document into chunks...")
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Document split into {len(chunks)} chunks, each with correct page number metadata.")

        return chunks

    except Exception as e:
        logger.error(f"Failed to load and split content: {e}")
        return []