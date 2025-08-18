# app/agent.py

import logging
from typing import Dict, Any, List

# --- MODIFIED: Import both Endpoint and Chat wrapper ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import BaseRetriever, Document
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
        Builds the complete RAG chain using LangChain Expression Language (LCEL).
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
                
        human_prompt_template = """
        CONTEXT:
        {context}

        QUESTION:
        {question}

        ASSISTANT:
        """
        
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            HumanMessagePromptTemplate.from_template(human_prompt_template)
        ])

        # 2. Initialize the Hugging Face LLM
        # First, create the core endpoint connection
        endpoint = HuggingFaceEndpoint(
            repo_id=self.llm_model_name,
            huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
            temperature=0.1,
            max_new_tokens=2048,
            return_full_text=False # Important for chat models
        )

        # Second, wrap the endpoint in the ChatHuggingFace class
        llm = ChatHuggingFace(llm=endpoint)

        # 3. Construct a more direct RAG chain
        rag_chain = (
            {
                "context": self.retriever | self._format_docs, 
                "question": RunnablePassthrough()
            }
            | chat_prompt
            | llm
            | StrOutputParser()
        )
        
        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "answer": rag_chain}
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

    def ask(self, query: str) -> Dict[str, Any]:
        logger.info(f"Received query: {query}")
        
        try:
            logger.info("Invoking RAG chain...")
            result = self.rag_chain.invoke(query)
            logger.info("RAG chain invocation complete.")

            raw_answer = result.get("answer", "")
            final_answer = raw_answer
            # Your original parsing logic for "Final Answer:" is kept
            if "Final Answer:" in raw_answer:
                parts = raw_answer.split("Final Answer:", 1)
                if len(parts) > 1:
                    final_answer = parts[1].strip()
            
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