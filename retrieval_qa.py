"""
Retrieval & QA Module for RAG System.

This module provides functionality for semantic search and question answering
using vector database retrieval and LLM-based answer generation. It integrates
with the storage_indexing module for vector database operations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
from storage_indexing import Document, StorageManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class representing a search result with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

@dataclass
class QAResponse:
    """Data class representing a QA response with answer and sources."""
    answer: str
    sources: List[SearchResult]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class RetrievalQA:
    """
    Main class for handling document retrieval and question answering.
    
    This class provides:
    1. Semantic search functionality using vector embeddings
    2. Document retrieval from vector database
    3. LLM-based answer generation with context
    """
    
    def __init__(
        self,
        storage_manager: StorageManager,
        model_name: str = "reader-lm:1.5b",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_k: int = 3
    ):
        """
        Initialize the RetrievalQA system.
        
        Args:
            storage_manager: Instance of StorageManager for vector DB operations
            model_name: Name of the LLM model to use
            max_tokens: Maximum tokens for LLM response
            temperature: LLM temperature parameter
            top_k: Number of documents to retrieve
        """
        self.storage_manager = storage_manager
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.llm = ollama.Client()
        
        logger.info(f"Initialized RetrievalQA with model: {model_name}")
        
    def retrieve_documents(
        self,
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: User's query string
            filter_criteria: Optional filtering criteria for the search
            
        Returns:
            List of SearchResult objects containing relevant documents
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Get similar documents from vector DB
        results = self.storage_manager.query_similar_documents(
            query=query,
            top_k=self.top_k,
            filter_criteria=filter_criteria
        )
        
        # Convert to SearchResult objects
        search_results = []
        for doc, score in results:
            search_result = SearchResult(
                content=doc.content,
                metadata=doc.metadata,
                similarity_score=score
            )
            search_results.append(search_result)
            
        logger.info(f"Retrieved {len(search_results)} documents")
        return search_results
        
    def _construct_prompt(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> str:
        """
        Construct a prompt combining the query and retrieved documents.
        
        Args:
            query: User's query
            documents: List of retrieved documents
            
        Returns:
            Formatted prompt string
        """
        # Start with system context
        prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Your answers should:
1. Be accurate and based on the given context
2. Be comprehensive yet concise
3. Include relevant quotes or references when appropriate
4. Acknowledge if the context doesn't fully answer the question

Context from relevant documents:

"""
        
        # Add retrieved documents
        for i, doc in enumerate(documents, 1):
            prompt += f"\nDocument {i}:\n"
            prompt += f"Title: {doc.metadata.get('title', 'Untitled')}\n"
            prompt += f"Content:\n{doc.content}\n"
            prompt += "-" * 80 + "\n"
            
        # Add the query
        prompt += f"\nQuestion: {query}\n\n"
        prompt += "Answer: "
        
        return prompt
        
    def generate_answer(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> QAResponse:
        """
        Generate an answer using the LLM based on retrieved documents.
        
        Args:
            query: User's query
            documents: List of retrieved relevant documents
            
        Returns:
            QAResponse object containing the answer and metadata
        """
        # Construct the prompt
        prompt = self._construct_prompt(query, documents)
        
        try:
            # Generate answer using LLM
            response = self.llm.generate(
                model=self.model_name,
                prompt=prompt
                # temperature=self.temperature,
                # max_tokens=self.max_tokens
            )
            
            # Create QA response
            qa_response = QAResponse(
                answer=response.response.strip(),
                sources=documents,
                prompt_tokens=response.prompt_eval_count,
                completion_tokens=response.eval_count,
                total_tokens=response.prompt_eval_count + response.eval_count
            )
            
            logger.info(
                f"Generated answer (tokens: {qa_response.total_tokens})"
            )
            return qa_response
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
            
    def query(
        self,
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> QAResponse:
        """
        Process a query through the complete retrieval-QA pipeline.
        
        This method:
        1. Retrieves relevant documents
        2. Generates an answer using the LLM
        
        Args:
            query: User's query string
            filter_criteria: Optional filtering criteria for document retrieval
            
        Returns:
            QAResponse object containing the answer and sources
        """
        # Retrieve relevant documents
        documents = self.retrieve_documents(
            query=query,
            filter_criteria=filter_criteria
        )
        
        if not documents:
            logger.warning("No relevant documents found")
            return QAResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
            
        # Generate answer
        return self.generate_answer(query, documents)

def create_retrieval_qa(
    storage_manager: StorageManager,
    **kwargs
) -> RetrievalQA:
    """
    Factory function to create a RetrievalQA instance.
    
    Args:
        storage_manager: StorageManager instance for vector DB operations
        **kwargs: Additional arguments for RetrievalQA initialization
        
    Returns:
        Configured RetrievalQA instance
    """
    return RetrievalQA(
        storage_manager=storage_manager,
        model_name=kwargs.get("model_name", "reader-lm:1.5b"),
        max_tokens=kwargs.get("max_tokens", 1000),
        temperature=kwargs.get("temperature", 0.6),
        top_k=kwargs.get("top_k", 3)
    )

def demo():
    """
    Demonstration of the Retrieval & QA module functionality.
    """
    from storage_indexing import create_storage_manager
    
    try:
        # Initialize storage manager (assuming Pinecone)
        storage_manager = create_storage_manager(
            vector_db_type="pinecone",
            api_key="your-api-key",  # Replace with actual key
            environment="your-environment",
            index_name="rag-demo"
        )
        
        # Create retrieval QA system
        qa_system = create_retrieval_qa(storage_manager)
        
        # Example query
        query = "What are the key features of vector databases?"
        
        # Get answer
        response = qa_system.query(query)
        
        # Print results
        print(f"\nQuery: {query}")
        print("\nAnswer:")
        print(response.answer)
        print("\nSources:")
        for i, source in enumerate(response.sources, 1):
            print(f"\nSource {i} (Score: {source.similarity_score:.3f}):")
            print(f"Title: {source.metadata.get('title', 'Untitled')}")
            print(f"Preview: {source.content[:200]}...")
            
        print(f"\nTokens used: {response.total_tokens}")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")

if __name__ == "__main__":
    demo() 