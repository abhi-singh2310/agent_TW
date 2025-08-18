# app/services/loaders.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging
import pypdf # Make sure pypdf is in your requirements.txt
from app import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _ensure_text_file_exists():
    """
    Checks if the text version of the PDF exists. If not, it creates it.
    This is a one-time preprocessing step that runs automatically.
    """
    text_file_path = config.DATA_DIR / "faq_manual.txt"
    pdf_file_path = config.PDF_FILE_PATH

    if text_file_path.exists():
        logger.info(f"Text file already exists, skipping PDF conversion.")
        return

    logger.info(f"Text file not found. Converting PDF to text...")
    try:
        with open(pdf_file_path, "rb") as pdf_file:
            reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        logger.info(f"Successfully converted PDF to text: {text_file_path}")

    except FileNotFoundError:
        logger.error(f"Error: The source PDF file was not found at {pdf_file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during PDF conversion: {e}")
        raise


def load_and_split_pdf(pdf_path: str = None) -> List[Document]:
    """
    Ensures a text version of the PDF exists, then loads and splits it into chunks.
    """
    try:
        # Step 1: Make sure the .txt file is available
        _ensure_text_file_exists()

        # Step 2: Load from the guaranteed .txt file
        text_file_path = config.DATA_DIR / "faq_manual.txt"
        logger.info(f"Loading content from pre-processed text file: {text_file_path}")

        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()

        pages = [Document(page_content=text, metadata={"source": "faq_manual.txt", "page": 0})]
        logger.info("Successfully loaded content.")

        # Step 3: Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=250,
            length_function=len
        )
        logger.info("Splitting document into chunks...")
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Document split into {len(chunks)} chunks.")

        return chunks

    except Exception as e:
        logger.error(f"Failed to load and split content: {e}")
        return []