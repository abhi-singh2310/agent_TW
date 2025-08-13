# app/services/loaders.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    Loads a PDF document from the given path and splits it into smaller chunks.

    This function uses PyPDFLoader to load the document page by page and then
    RecursiveCharacterTextSplitter to split the text into chunks of a specified
    size with some overlap. This overlap helps maintain context between chunks.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        List[Document]: A list of Document objects, where each object represents
                        a chunk of the original document. The metadata of each
                        chunk includes the source and page number.
    """
    try:
        logger.info(f"Loading PDF from path: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logger.info(f"Successfully loaded {len(pages)} pages from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,  # The maximum size of a chunk (in characters)
            chunk_overlap=250, # The number of characters to overlap between chunks
            length_function=len
        )

        logger.info("Splitting document into chunks...")
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Document split into {len(chunks)} chunks.")

        return chunks

    except Exception as e:
        logger.error(f"Failed to load and split PDF from {pdf_path}: {e}")
        return []