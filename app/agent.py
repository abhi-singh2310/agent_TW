# app/agent.py

import logging
from typing import Dict, Any, List
from operator import itemgetter
import re

# --- MODIFIED: Import both Endpoint and Chat wrapper ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import BaseRetriever, Document
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from requests.exceptions import HTTPError
from app import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenAIAgent:
    """
    The core agent class that orchestrates the RAG pipeline.
    """
    def __init__(self, retriever: BaseRetriever, llm_model_name: str):
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.rag_chain = self._build_rag_chain()
        logger.info(f"GenAIAgent initialized with retriever and LLM: {llm_model_name}")

    def _build_rag_chain(self):
        """
        Builds the complete RAG chain using LangChain Expression Language (LCEL) with memory.
        """
        # 1. Define the prompt using a structured chat template
        system_prompt_template = """
        You are a helpful customer support assistant. Your task is to answer the user's question based ONLY on the provided context.
        Follow the user's instructions and the examples below precisely.

        Example:
        CONTEXT:
        [Context about returns: "Most items can be returned within 30 days. The customer is responsible for return shipping costs for change-of-mind returns. Custom-made furniture cannot be returned."]
        QUESTION:
        Can I return a custom-made sofa if I don't like it?
        ASSISTANT:
        Relevant Information:
        1. Custom-made furniture cannot be returned.

        Final Answer:
        I'm sorry, but custom-made furniture, including sofas, cannot be returned for a change of mind.
        """
        
        # We use MessagesPlaceholder to inject the chat history
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nASSISTANT:"),
            ]
        )

        # 2. Initialize the Hugging Face LLM
        endpoint = HuggingFaceEndpoint(
            repo_id=self.llm_model_name,
            huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
            temperature=0.1,
            max_new_tokens=2048,
            return_full_text=False
        )
        llm = ChatHuggingFace(llm=endpoint)

        # 3. Construct a more direct RAG chain
        rag_chain = (
            RunnableParallel(
                {
                    # Pass the question to the retriever
                    "context": itemgetter("question") | self.retriever | self._format_docs,
                    # Pass the original inputs for the prompt
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        rag_chain_with_source = RunnableParallel(
            {
                # Pass only the question to the retriever for the source context
                "context": itemgetter("question") | self.retriever,
                "answer": rag_chain
            }
        )

        return rag_chain_with_source

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    @staticmethod
    def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        if not docs:
            return []
        
        sources = [
            {"source": doc.metadata.get('source', 'unknown'), "page": doc.metadata.get('page', 'unknown')}
            for doc in docs
        ]
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        return sorted(unique_sources, key=lambda x: x.get('page', 0))

    # New method to format the final answer for readability
    def _format_final_answer(self, text: str) -> str:
        # Remove leading/trailing whitespace
        text = text.strip()
        # Ensure newlines for numbered lists
        text = re.sub(r'(\d+\.)', r'\n\n\1', text)
        # Ensure newlines for bullet points
        text = re.sub(r'(\n•)', r'\n\n•', text)
        return text.strip()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError)
    )
    def ask(self, query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        logger.info(f"Received query: {query}")
        
        # Format the history for the chat model
        if history is None:
            history = []
        
        # We need to correctly format the history to work with MessagesPlaceholder
        formatted_history = []
        for msg in history:
            if msg['sender'] == 'user':
                formatted_history.append(("human", msg['text']))
            elif msg['sender'] == 'bot':
                formatted_history.append(("ai", msg['text']))

        try:
            logger.info("Invoking RAG chain...")
            # Pass both the query and formatted history to the chain
            result = self.rag_chain.invoke({"question": query, "chat_history": formatted_history})
            logger.info("RAG chain invocation complete.")

            raw_answer = result.get("answer", "")
            final_answer = raw_answer
            # Your original parsing logic for "Final Answer:" is kept
            if "Final Answer:" in raw_answer:
                parts = raw_answer.split("Final Answer:", 1)
                if len(parts) > 1:
                    final_answer = parts[1].strip()
            
            # Apply the new formatting
            final_answer = self._format_final_answer(final_answer)

            source_docs = result.get("context", [])
            formatted_sources = self._format_sources(source_docs)

            response = {
                "answer": final_answer,
                "sources": formatted_sources
            }
            logger.info("Successfully processed query and generated response.")
            return response
            
        except Exception as e:
            logger.error(f"An error occurred during RAG chain invocation: {e}", exc_info=True)
            return {
                "answer": "Sorry, I encountered an error. Please check the server logs for details.",
                "sources": []
            }