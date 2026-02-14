"""Retrieval strategies for Abhinav Digital Twin RAG system"""

from typing import List, Literal
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.rag.utils import get_vectorstore, get_chroma_client


def get_all_documents_from_collection(collection_name: str) -> List[Document]:
    """Retrieve all documents from a Chroma collection.
    
    Args:
        collection_name: Name of collection to retrieve from
        
    Returns:
        List of Document objects
    """
    vectorstore = get_vectorstore(collection_name)
    
    # Get all documents from the collection
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    results = collection.get(include=["documents", "metadatas"])
    
    # Convert to LangChain Document objects
    documents = []
    for i, doc_text in enumerate(results['documents']):
        metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
        documents.append(Document(page_content=doc_text, metadata=metadata))
    
    return documents


def get_semantic_retriever(
    collection_name: str,
    k: int = 3,
    search_type: str = "similarity"
) -> BaseRetriever:
    """Get semantic (ANN) retriever using cosine similarity.
    
    Args:
        collection_name: Name of collection to search
        k: Number of documents to retrieve
        search_type: Type of search (similarity, mmr)
        
    Returns:
        Chroma retriever instance
    """
    vectorstore = get_vectorstore(collection_name)
    
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    
    return retriever


def get_retriever(
    collection: str,
    strategy: Literal["semantic"] = "semantic",
    k: int = 3
) -> BaseRetriever:
    """Get retriever based on strategy and collection.
    
    For now, only semantic search is supported. BM25 and hybrid can be added later.
    
    Args:
        collection: Collection name ("personal", "professional", or "both")
        strategy: Retrieval strategy to use (currently only "semantic")
        k: Number of documents to retrieve per collection
        
    Returns:
        Appropriate retriever instance
    """
    if strategy != "semantic":
        print(f"Warning: Only 'semantic' strategy is currently supported. Using semantic search.")
    
    # For "both", we'll retrieve from both collections separately
    if collection == "both":
        # Return a custom retriever that queries both
        return MultiCollectionRetriever(
            collections=["personal", "professional"],
            k=k
        )
    else:
        return get_semantic_retriever(collection, k=k)


class MultiCollectionRetriever(BaseRetriever):
    """Custom retriever that queries multiple collections."""
    
    collections: List[str]
    k: int = 3
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve from all collections and combine results."""
        all_docs = []
        
        for collection in self.collections:
            retriever = get_semantic_retriever(collection, k=self.k)
            docs = retriever.invoke(query)
            all_docs.extend(docs)
        
        # Return top k overall by taking first k from combined results
        return all_docs[:self.k * len(self.collections)]


def retrieve_documents(
    query: str,
    collection: str,
    strategy: Literal["semantic"] = "semantic",
    k: int = 3
) -> List[Document]:
    """Retrieve documents using specified strategy.
    
    Args:
        query: Query string
        collection: Collection to search
        strategy: Retrieval strategy (currently only semantic)
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved Document objects
    """
    retriever = get_retriever(collection, strategy, k)
    documents = retriever.invoke(query)
    
    return documents
