"""
RAG System Package
A production-ready Retrieval-Augmented Generation system.
"""

from .ingest import DocumentIngestor, PDFIngestor
from .query import RAGQueryEngine
from .rag import RAGSystem

__all__ = ["DocumentIngestor", "PDFIngestor", "RAGQueryEngine", "RAGSystem"]
__version__ = "1.0.0"
