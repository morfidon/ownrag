"""
Document Ingestion Module for RAG System
Handles PDF, TXT, and other document parsing, chunking, embedding generation, and vector storage.
Tracks ingested files to avoid re-processing.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class DocumentIngestor:
    """Handles multi-format document ingestion into ChromaDB vector store with file tracking."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the document ingestor.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        load_dotenv()
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.persist_directory = persist_directory
        self.tracking_file = os.path.join(persist_directory, "ingested_files.json")
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.api_key
        )
        
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _load_tracking_data(self) -> Dict[str, str]:
        """Load tracking data of ingested files."""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_tracking_data(self, tracking_data: Dict[str, str]) -> None:
        """Save tracking data of ingested files."""
        os.makedirs(self.persist_directory, exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    
    def _is_file_ingested(self, file_path: str) -> bool:
        """Check if file has already been ingested and hasn't changed."""
        tracking_data = self._load_tracking_data()
        file_path = str(Path(file_path).resolve())
        
        if file_path not in tracking_data:
            return False
        
        current_hash = self._get_file_hash(file_path)
        return tracking_data[file_path] == current_hash
    
    def _mark_file_ingested(self, file_path: str) -> None:
        """Mark file as ingested with its current hash."""
        tracking_data = self._load_tracking_data()
        resolved_path = str(Path(file_path).resolve())
        
        # Check if file exists before hashing
        if not os.path.exists(resolved_path):
            print(f"Warning: File not found for tracking: {resolved_path}")
            return
        
        file_hash = self._get_file_hash(resolved_path)
        tracking_data[resolved_path] = file_hash
        self._save_tracking_data(tracking_data)
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on file extension.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of Document objects with content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext in {'.txt', '.md'}:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: {self.SUPPORTED_EXTENSIONS}")
        
        documents = loader.load()
        
        # Add source file to metadata
        for doc in documents:
            doc.metadata['source_file'] = Path(file_path).name
            doc.metadata['file_type'] = file_ext
        
        print(f"Loaded {len(documents)} page(s)/section(s) from {Path(file_path).name}")
        return documents
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        Split documents into chunks with overlap.
        
        Args:
            documents: List of Document objects
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        return chunks
    
    def create_or_update_vector_store(self, chunks: List[Document]) -> Chroma:
        """
        Create or update ChromaDB vector store with document chunks.
        
        Args:
            chunks: List of chunked Document objects
            
        Returns:
            Chroma vector store instance
        """
        print(f"Generating embeddings and storing in ChromaDB...")
        
        if os.path.exists(self.persist_directory):
            # Load existing and add new chunks
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            vectorstore.add_documents(chunks)
            print(f"Added {len(chunks)} chunks to existing vector store")
        else:
            # Create new vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Created new vector store with {len(chunks)} chunks")
        
        print(f"Persisted to: {self.persist_directory}")
        return vectorstore
    
    def ingest_file(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50, force: bool = False) -> Optional[Chroma]:
        """
        Ingest a single file: load → chunk → embed → store.
        Skips if already ingested unless force=True.
        
        Args:
            file_path: Path to file
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            force: Force re-ingestion even if file already processed
            
        Returns:
            Chroma vector store instance or None if skipped
        """
        file_path = str(Path(file_path).resolve())
        
        # Check if already ingested
        if not force and self._is_file_ingested(file_path):
            print(f"Skipping {Path(file_path).name} (already ingested)")
            return None
        
        print(f"\nIngesting: {Path(file_path).name}")
        
        # Load document
        documents = self.load_document(file_path)
        
        # Chunk documents
        chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
        
        # Create/update vector store
        vectorstore = self.create_or_update_vector_store(chunks)
        
        # Mark as ingested
        self._mark_file_ingested(file_path)
        
        print(f"Ingestion complete.\n")
        
        return vectorstore
    
    def ingest_folder(self, folder_path: str, chunk_size: int = 500, chunk_overlap: int = 50, 
                     extensions: Optional[Set[str]] = None, force: bool = False) -> Chroma:
        """
        Ingest all supported files in a folder.
        Only processes new or modified files unless force=True.
        
        Args:
            folder_path: Path to folder containing documents
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            extensions: Set of file extensions to process (default: all supported)
            force: Force re-ingestion of all files
            
        Returns:
            Chroma vector store instance
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        extensions = extensions or self.SUPPORTED_EXTENSIONS
        
        print(f"\nScanning folder: {folder_path}")
        
        # Find all supported files
        files_to_process = []
        for ext in extensions:
            files_to_process.extend(folder_path.glob(f"*{ext}"))
        
        if not files_to_process:
            print(f"No supported files found in {folder_path}")
            print(f"   Supported extensions: {extensions}")
            return self.load_existing_vectorstore()
        
        print(f"Found {len(files_to_process)} file(s): {[f.name for f in files_to_process]}\n")
        
        # Ingest each file
        ingested_count = 0
        skipped_count = 0
        
        for file_path in files_to_process:
            result = self.ingest_file(str(file_path), chunk_size, chunk_overlap, force)
            if result is not None:
                ingested_count += 1
            else:
                skipped_count += 1
        
        print(f"\nFolder ingestion summary:")
        print(f"  Ingested: {ingested_count} file(s)")
        print(f"  Skipped: {skipped_count} file(s) (already processed)\n")
        
        return self.load_existing_vectorstore()
    
    def load_existing_vectorstore(self) -> Chroma:
        """
        Load existing ChromaDB vector store from disk.
        
        Returns:
            Chroma vector store instance
            
        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run ingestion first."
            )
        
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        print(f"Loaded existing vector store from {self.persist_directory}")
        return vectorstore
    
    def get_ingested_files(self) -> Dict[str, str]:
        """Get list of all ingested files with their hashes."""
        return self._load_tracking_data()
    
    def clear_tracking(self) -> None:
        """Clear file tracking data (useful for forcing re-ingestion)."""
        if os.path.exists(self.tracking_file):
            os.remove(self.tracking_file)
            print("Cleared file tracking data")


# Backward compatibility alias
PDFIngestor = DocumentIngestor


def main():
    """Example usage of DocumentIngestor."""
    # Initialize ingestor
    ingestor = DocumentIngestor(persist_directory="./chroma_db")
    
    # Ingest entire folder
    vectorstore = ingestor.ingest_folder("./data")
    
    # Show ingested files
    print("\nIngested files:")
    for file_path in ingestor.get_ingested_files():
        print(f"  - {Path(file_path).name}")


if __name__ == "__main__":
    main()
