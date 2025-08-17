# app/agent.py

import logging
import json
from typing import Dict, Any, List
from operator import itemgetter

from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import BaseRetriever, Document

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

        Args:
            retriever (BaseRetriever): The document retriever instance.
            llm_model_name (str): The name of the Ollama model to use for generation.
        """
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.rag_chain = self._build_rag_chain()
        logger.info(f"GenAIAgent initialized with retriever and LLM: {llm_model_name}")

    def _build_rag_chain(self):
        """
        Builds the complete RAG chain using LangChain Expression Language (LCEL).
        """
        # 1. Define the prompt template
        template = """
            **Role:** You are a helpful assistant for a customer support team.

            **Task:** Your goal is to provide accurate and helpful answers to customer questions based *only* on the provided context. You will be given a few examples of how to answer questions.

            **Instructions:**
            1.  **Follow the Examples:** Emulate the tone and style of the examples provided below.
            2.  **Analyze the Context:** Carefully read the retrieved context before answering the new question.
            3.  **Synthesize the Answer:** Formulate a clear and concise answer to the user's question using only the information from the context.
            4.  **Constraints:**
                * Do not use any information outside of the provided context.
                * If the answer is not in the context, explicitly state: "I'm sorry, but I don't have enough information to answer that question."
                * Keep your answer to a maximum of three sentences.

            ---
            **Examples:**

            **Example 1:**
            **Question:** What tools will I need for assembly?
            [cite_start]**Answer:** All the tools you need, which is usually an Allen key, are provided in the box. [cite: 11] [cite_start]For some of our more complex items, like modular sofas or beds, you might also need a standard Phillips-head screwdriver. [cite: 12]

            **Example 2:**
            **Question:** Can I return a sofa if I change my mind?
            [cite_start]**Answer:** You can return most items within 30 days of delivery, as long as they are unused and in their original packaging. [cite: 24, 25] [cite_start]However, for change-of-mind returns, the cost of the return shipping will be deducted from your refund. [cite: 28]

            **Example 3:**
            **Question:** How do I take care of my new wooden table?
            [cite_start]**Answer:** To care for your wooden furniture, you should dust it regularly with a soft cloth and avoid placing it in direct sunlight. [cite: 56, 57] [cite_start]It's also a good idea to use coasters to prevent marks and apply wood polish every 6 to 12 months. [cite: 58, 59]

            ---

            **New Task:**

            **Context:**
            {context}

            **Question:**
            {question}

            **Answer:**
        """
        prompt = PromptTemplate.from_template(template)

        # 2. Initialize the LLM
        llm = ChatOllama(model=self.llm_model_name, temperature=0.1)

        # 3. Construct the RAG chain
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
        # Remove duplicate sources
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        return sorted(unique_sources, key=lambda x: x.get('page', 0))

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Executes a query against the RAG chain and returns a single result.
        """
        logger.info(f"Received query: {query}")
        
        # Use .invoke() which returns the final result
        result = self.rag_chain.invoke(query)
        
        # Format the sources from the context
        source_docs = result.get("context", [])
        formatted_sources = self._format_sources(source_docs)
        
        # Combine the answer and sources into a single response
        response = {
            "answer": result.get("answer"),
            "sources": formatted_sources
        }
        
        return response
