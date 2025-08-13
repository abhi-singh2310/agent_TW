# app/agent.py

import logging
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
        You are a helpful assistant for a customer support team.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer from the provided context, clearly state that you don't know.
        Your answer should be concise, professional, and limited to a maximum of three sentences.

        Context:
        {context}

        Question:
        {question}

        Answer:
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
        Executes a query against the RAG chain and returns the answer and sources.

        Args:
            query (str): The user's question.

        Returns:
            Dict[str, Any]: A dictionary containing the 'answer' and 'sources'.
        """
        logger.info(f"Received query: {query}")
        result = self.rag_chain.invoke(query)
        
        answer = result.get("answer", "No answer could be generated.")
        source_docs = result.get("context", [])
        
        formatted_sources = self._format_sources(source_docs)

        logger.info(f"Generated answer: {answer}")
        logger.info(f"Sources: {formatted_sources}")

        return {
            "answer": answer,
            "sources": formatted_sources
        }