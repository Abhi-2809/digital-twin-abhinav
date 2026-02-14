"""Utility functions for Abhinav Digital Twin RAG system"""

import os
import chromadb
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()


def get_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """Get OpenAI embeddings model.
    
    Args:
        model: OpenAI embedding model name
        
    Returns:
        OpenAIEmbeddings instance
    """
    return OpenAIEmbeddings(model=model)


def get_llm(model: str = "gpt-4.1-nano", temperature: float = 0.0) -> ChatOpenAI:
    """Get ChatOpenAI LLM instance.
    
    Args:
        model: OpenAI model name
        temperature: Sampling temperature (0.0 for deterministic)
        
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(model=model, temperature=temperature)


def get_chroma_client(persist_directory: str = "./vector_store") -> chromadb.PersistentClient:
    """Get ChromaDB persistent client.
    
    Args:
        persist_directory: Path to vector store
        
    Returns:
        ChromaDB PersistentClient instance
    """
    return chromadb.PersistentClient(path=persist_directory)


def get_vectorstore(
    collection_name: str,
    persist_directory: str = "./vector_store"
) -> Chroma:
    """Get Chroma vectorstore for a specific collection.
    
    Args:
        collection_name: Name of the collection ("personal" or "professional")
        persist_directory: Path to vector store
        
    Returns:
        Chroma vectorstore instance
    """
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore


def validate_api_key() -> bool:
    """Validate that OpenAI API key is set.
    
    Returns:
        True if API key exists, False otherwise
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please ensure .env file exists with valid API key."
        )
    return True
