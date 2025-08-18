# app/agent.py

import logging
from typing import Dict, Any, List

# --- MODIFIED: Import HuggingFaceEndpoint instead of ChatOllama ---
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import BaseRetriever, Document
from app import config # Import the config module to access the API key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenAIAgent:
    """
    The core agent class that orchestrates the RAG pipeline.
    """
    def __init__(self, retriever: BaseRetriever, llm_model_name: str):
        """
        Initializes the agent with a retriever and the specified LLM.
        """
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.rag_chain = self._build_rag_chain()
        logger.info(f"GenAIAgent initialized with retriever and LLM: {llm_model_name}")

    def _build_rag_chain(self):
        """
        Builds the complete RAG chain using LangChain Expression Language (LCEL).
        """
        # 1. Define the prompt template (This remains unchanged)
        template = """
            <|system|>
            You are a helpful customer support assistant. Your task is to answer the user's question based ONLY on the provided context.
            Follow the user's instructions and the examples below precisely.
            </s>
            <|user|>
            CONTEXT:
            [Context about returns: "Most items can be returned within 30 days. The customer is responsible for return shipping costs for change-of-mind returns. Custom-made furniture cannot be returned."]

            QUESTION:
            Can I return a custom-made sofa if I don't like it?

            </s>
            <|assistant|>
            Relevant Information:
            1. Custom-made furniture cannot be returned.

            Final Answer:
            I'm sorry, but custom-made furniture, including sofas, cannot be returned for a change of mind.
            <|user|>
            CONTEXT:
            {context}

            QUESTION:
            {question}
            </s>
            <|assistant|>
        """
        prompt = PromptTemplate.from_template(template)

        # --- MODIFIED: Initialize the Hugging Face LLM ---
        # 2. Initialize the LLM using HuggingFaceEndpoint instead of ChatOllama
        llm = HuggingFaceEndpoint(
            repo_id=self.llm_model_name,
            huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
            temperature=0.1,
            max_new_tokens=1024, # Controls the maximum length of the response
        )

        # 3. Construct the RAG chain (This remains unchanged)
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self._format_docs(x["context"]))
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

    # ... The rest of your methods (_format_docs, _format_sources, ask) are unchanged ...
    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        Formats the retrieved documents into a single string for the prompt.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    @staticmethod
    def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Formats the source documents into a structured list for the final output.
        """
        if not docs:
            return []
        
        sources = [
            {
                "source": doc.metadata.get('source', 'unknown'),
                "page": doc.metadata.get('page', 'unknown')
            }
            for doc in docs
        ]
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        return sorted(unique_sources, key=lambda x: x.get('page', 0))

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Executes a query against the RAG chain and returns the final answer.
        """
        logger.info(f"Received query: {query}")
        result = self.rag_chain.invoke(query)
        raw_answer = result.get("answer", "")
        final_answer = raw_answer
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
        return response