"""ChromaDB Ingestion Script for Abhinav Digital Twin.
Loads PDFs from personal/professional folders into separate Chroma collections.
"""

import re
import chromadb
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()


def clean_text(text: str) -> str:
    """Remove special characters while keeping numbers, dates, and periods.
    
    Args:
        text: Raw text from PDF page
        
    Returns:
        Cleaned text with only alphanumeric, spaces, and periods
    """
    # Remove special chars but keep word chars, spaces, digits, and periods
    cleaned = re.sub(r'[^\w\s\d\.]', ' ', text)
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def process_documents(docs: List, category: str) -> List:
    """Clean and chunk documents with metadata.
    
    Args:
        docs: List of Document objects from DirectoryLoader
        category: "personal" or "professional"
        
    Returns:
        List of chunked Document objects with metadata
    """
    # Clean text in each document
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    chunks = text_splitter.split_documents(docs)
    
    # Add metadata to each chunk
    for chunk in chunks:
        # Extract filename from source path
        source_path = Path(chunk.metadata.get('source', ''))
        filename = source_path.name
        
        # Add/update metadata
        chunk.metadata['category'] = category
        chunk.metadata['source'] = filename
        # page number already exists in metadata from PyPDFLoader
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = 0
    
    return chunks


def create_collection(client: chromadb.PersistentClient, 
                      collection_name: str, 
                      chunks: List) -> int:
    """Create or get Chroma collection and add document chunks.
    
    Args:
        client: ChromaDB PersistentClient
        collection_name: Name of collection to create
        chunks: List of chunked documents to add
        
    Returns:
        Number of chunks added to collection
    """
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./vector_store"
    )
    
    return len(chunks)


def ingest_data():
    """Main ingestion function to load PDFs into ChromaDB collections."""
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./vector_store")
    
    # Personal collection
    print("Loading personal documents...")
    personal_path = "./data/personal/"
    if not Path(personal_path).exists() or not list(Path(personal_path).glob("*.pdf")):
        print(f"No PDFs found in {personal_path}")
        personal_chunks = []
        personal_doc_count = 0
    else:
        personal_loader = DirectoryLoader(
            personal_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        personal_docs = personal_loader.load()
        personal_doc_count = len(set(doc.metadata['source'] for doc in personal_docs))
        
        # Process documents
        personal_chunks = process_documents(personal_docs, "personal")
        
        # Create collection
        create_collection(client, "personal", personal_chunks)
    
    # Professional collection
    print("Loading professional documents...")
    professional_path = "./data/professional/"
    if not Path(professional_path).exists() or not list(Path(professional_path).glob("*.pdf")):
        print(f"⚠️ No PDFs found in {professional_path}")
        professional_chunks = []
        professional_doc_count = 0
    else:
        professional_loader = DirectoryLoader(
            professional_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        professional_docs = professional_loader.load()
        professional_doc_count = len(set(doc.metadata['source'] for doc in professional_docs))
        
        # Process documents
        professional_chunks = process_documents(professional_docs, "professional")
        
        # Create collection
        create_collection(client, "professional", professional_chunks)
    
    # Print statistics
    print("\n" + "="*50)
    print("INGESTION COMPLETE")
    print("="*50)
    print(f"Personal collection: {personal_doc_count} docs, {len(personal_chunks)} chunks")
    print(f"Professional collection: {professional_doc_count} docs, {len(professional_chunks)} chunks")
    print("="*50)


if __name__ == "__main__":
    ingest_data()
