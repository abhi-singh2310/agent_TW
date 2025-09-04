# app/agent.py

"""
This module defines the core GenAIAgent class, which is the heart of the
GenAI-powered FAQ and document search agent.

The agent is built using the LangChain framework and follows a
Retrieval-Augmented Generation (RAG) architecture. This approach was chosen
as it's highly effective for question-answering tasks over a specific body of
knowledge. The RAG pipeline ensures that the agent's responses are grounded in 
the provided documentation (FAQ), minimising the risk of hallucinations and 
providing customers with accurate, context-aware answers.
"""

import logging
import re
from operator import itemgetter
from typing import Any, Dict, List

from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.schema import BaseRetriever, Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from requests.exceptions import HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app import config

# --- Logging Configuration ---
# We configure a logger to provide insights into the agent's operations,
# which is crucial for debugging and monitoring in a real-world application.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenAIAgent:
    """
    The GenAIAgent orchestrates the entire RAG pipeline.

    This class is responsible for initialising the language model (LLM),
    constructing the prompt template, and building the LangChain Expression
    Language (LCEL) chain that connects all the components. The design is
    modular, allowing for easier testing and potential future enhancements,
    such as swapping out the retriever or LLM.

    Attributes:
        retriever (BaseRetriever): The retriever object responsible for
            fetching relevant documents from the vector store.
        llm_model_name (str): The name of the Hugging Face model to be used.
        rag_chain (Runnable): The complete, executable RAG chain.
    """

    def __init__(self, retriever: BaseRetriever, llm_model_name: str):
        """
        Initialises the GenAIAgent.

        Args:
            retriever (BaseRetriever): An initialised retriever object.
            llm_model_name (str): The repository ID of the Hugging Face model.
        """
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.rag_chain = self._build_rag_chain()
        logger.info(
            "GenAIAgent initialised with retriever and LLM: %s",
            llm_model_name
        )

    def _build_rag_chain(self):
        """
        Builds the RAG chain using LangChain Expression Language (LCEL).

        The chain is constructed to be memory-aware, meaning it can consider
        past conversation history. This is crucial for a customer support
        chatbot to handle follow-up questions effectively.

        The chain performs the following steps:
        1.  Retrieves context relevant to the user's question.
        2.  Constructs a detailed prompt including the retrieved context,
            chat history, and the user's question.
        3.  Invokes the LLM to generate an answer.
        4.  Parses the LLM's output to a string.
        5.  Simultaneously retrieves the source documents to be returned
            alongside the answer.

        Returns:
            Runnable: The complete, executable RAG chain.
        """
        # 1. Define the prompt template. This is a critical part of the
        #    RAG pipeline, as it guides the LLM's behaviour. The template
        #    is designed to be robust, instructing the model to answer
        #    based *only* on the provided context. This is a key strategy
        #    to prevent the model from making things up.
        system_prompt_template = """
        You are a helpful customer support assistant for Temple & Webster.
        Your task is to answer the user's question based ONLY on the
        provided context. Follow the user's instructions and the examples
        below precisely.

        Example:
        CONTEXT:
        [Context about returns: "Most items can be returned within 30 days.
        The customer is responsible for return shipping costs for
        change-of-mind returns. Custom-made furniture cannot be returned."]
        QUESTION:
        Can I return a custom-made sofa if I don't like it?
        ASSISTANT:
        Relevant Information:
        1. Custom-made furniture cannot be returned.

        Final Answer:
        I'm sorry, but custom-made furniture, including sofas, cannot be
        returned for a change of mind.
        """

        # The `MessagesPlaceholder` is used to inject the chat history
        # into the prompt, making the agent conversational.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human",
             "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nASSISTANT:"),
        ])

        # 2. Initialise the Hugging Face LLM. We use `HuggingFaceEndpoint`
        #    as it's a lightweight way to interact with models hosted on the
        #    Hugging Face Hub. The `ChatHuggingFace` wrapper provides a
        #    standardised interface for chat-based models.
        endpoint = HuggingFaceEndpoint(
            repo_id=self.llm_model_name,
            huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
            temperature=0.1,
            max_new_tokens=2048,
            return_full_text=False)
        llm = ChatHuggingFace(llm=endpoint)

        # 3. Construct the RAG chain using LCEL. The `RunnableParallel`
        #    allows different parts of the chain to be executed in parallel,
        #    which is efficient.
        rag_chain = (RunnableParallel({
            "context":
            itemgetter("question") | self.retriever | self._format_docs,
            "question":
            itemgetter("question"),
            "chat_history":
            itemgetter("chat_history"),
        }) | prompt | llm | StrOutputParser())

        # This final chain runs the main RAG chain and simultaneously
        # retrieves the source documents to be returned with the answer.
        rag_chain_with_source = RunnableParallel({
            "context": itemgetter("question") | self.retriever,
            "answer": rag_chain
        })

        return rag_chain_with_source

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        Formats a list of documents into a single string.

        This is a helper function to prepare the retrieved documents for
        inclusion in the prompt.

        Args:
            docs (List[Document]): A list of LangChain Document objects.

        Returns:
            str: A single string with the content of all documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _extract_supporting_quotes(answer: str, docs: List[Document], max_quote_length: int = 120) -> Dict[str, str]:
        """
        Extracts relevant quotes from Q&A structured documents.
        
        Simple approach: Extract A: sections from Q&A pairs in reranked documents.
        
        Args:
            docs (List[Document]): The source documents used for answer generation.
            max_quote_length (int): Maximum length for extracted quotes.
            
        Returns:
            Dict[str, str]: Mapping of document identifier to relevant quote.
        """
        if not docs:
            return {}
        
        quotes = {}
        
        for doc in docs:
            doc_id = f"{doc.metadata.get('source', 'unknown')}_page_{doc.metadata.get('page', 'unknown')}"
            content = doc.page_content
            
            # Look for Q&A structure and extract answer section
            qa_match = re.search(r'Q:\s*[^\n]+\s*\n+A:\s*([^Q]+?)(?=Q:|$)', content, re.DOTALL)
            
            if qa_match:
                # Extract the A: section
                quote = qa_match.group(1).strip()
                # Clean up formatting while preserving structure
                quote = re.sub(r'\s+', ' ', quote)
                
                # Truncate if needed
                if len(quote) > max_quote_length:
                    quote = quote[:max_quote_length].rsplit(' ', 1)[0] + "..."
                    
                quotes[doc_id] = quote
                    
        return quotes

    @staticmethod
    def _format_sources(docs: List[Document], answer: str = "") -> List[Dict[str, Any]]:
        """
        Formats the source documents into a structured list with supporting quotes.

        This method extracts the source and page number from the metadata of
        each document, removes duplicates, sorts them, and includes supporting
        quotes from the actual content used.

        Args:
            docs (List[Document]): A list of LangChain Document objects.
            answer (str): The generated answer to find supporting quotes for.

        Returns:
            List[Dict[str, Any]]: A list of unique, sorted source dictionaries with quotes.
        """
        if not docs:
            return []

        # Extract supporting quotes for better transparency
        quotes = GenAIAgent._extract_supporting_quotes(answer, docs)
        
        sources = []
        seen_pages = set()
        
        for doc in docs:
            source_name = doc.metadata.get('source', 'unknown')
            page_num = doc.metadata.get('page', 'unknown')
            doc_id = f"{source_name}_page_{page_num}"
            
            # Avoid duplicate pages (in case multiple chunks from same page)
            page_key = (source_name, page_num)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            
            source_entry = {
                "source": source_name,
                "page": page_num,
                "quote": quotes.get(doc_id, "")
            }
            sources.append(source_entry)
            
        return sorted(sources, key=lambda x: x.get('page', 0))

    def _format_final_answer(self, text: str) -> str:
        """
        Applies final formatting to the LLM's answer for better readability.

        This includes trimming whitespace and ensuring proper spacing for
        lists, which improves the user experience of the chatbot.

        Args:
            text (str): The raw text output from the LLM.

        Returns:
            str: The formatted text.
        """
        text = text.strip()
        # These regex substitutions add newlines before list items, which
        # the LLM sometimes omits.
        text = re.sub(r'(\d+\.)', r'\n\n\1', text)
        text = re.sub(r'(\n•)', r'\n\n•', text)
        return text.strip()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type(HTTPError))
    def ask(self,
            query: str,
            history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Processes a user's query through the RAG chain.

        This is the main entry point for interacting with the agent. It takes
        a query and optional chat history, invokes the RAG chain, and
        formats the final response.

        The `@retry` decorator from the `tenacity` library is used to make
        the agent more resilient to transient network errors when calling
        the Hugging Face API.

        Args:
            query (str): The user's question.
            history (List[Dict[str, str]], optional): The conversation
                history. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and sources.
        """
        logger.info("Received query: %s", query)

        if history is None:
            history = []

        # The chat history needs to be formatted into a list of tuples
        # (e.g., `[("human", "hello"), ("ai", "hi")]`) for the prompt template.
        formatted_history = []
        for msg in history:
            if msg['sender'] == 'user':
                formatted_history.append(("human", msg['text']))
            elif msg['sender'] == 'bot':
                formatted_history.append(("ai", msg['text']))

        try:
            logger.info("Invoking RAG chain...")
            result = self.rag_chain.invoke({
                "question": query,
                "chat_history": formatted_history
            })
            logger.info("RAG chain invocation complete.")

            raw_answer = result.get("answer", "")
            # The prompt instructs the model to use "Final Answer:", so we
            # parse the output to extract only the final answer text.
            if "Final Answer:" in raw_answer:
                parts = raw_answer.split("Final Answer:", 1)
                if len(parts) > 1:
                    final_answer = parts[1].strip()
                else:
                    final_answer = raw_answer
            else:
                final_answer = raw_answer

            final_answer = self._format_final_answer(final_answer)
            source_docs = result.get("context", [])
            formatted_sources = self._format_sources(source_docs, final_answer)

            return {"answer": final_answer, "sources": formatted_sources}

        except Exception as e:
            logger.error("An error occurred during RAG chain invocation: %s",
                         e,
                         exc_info=True)
            return {
                "answer":
                ("Sorry, I encountered an error. Please check the server "
                 "logs for details."),
                "sources": []
            }
