# RAG System with ChromaDB and OpenAI

A production-ready Retrieval-Augmented Generation (RAG) system for querying PDF documents using ChromaDB vector database and OpenAI's GPT models.

## Features

- **PDF Ingestion**: Automatically parse, chunk, and embed PDF documents
- **Vector Storage**: Persistent ChromaDB vector database with efficient similarity search
- **Smart Retrieval**: Retrieve top-k most relevant chunks for each query
- **OpenAI Integration**: Uses `text-embedding-3-small` for embeddings and `gpt-4o-mini` for generation
- **Source Citations**: Answers include page number references
- **Modular Design**: Separate modules for ingestion, querying, and main interface

## Architecture

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  ingest.py          │
│  - Load PDF         │
│  - Chunk text       │
│  - Generate embeds  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   ChromaDB          │
│   (Vector Store)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  query.py           │
│  - Embed question   │
│  - Retrieve chunks  │
│  - Generate answer  │
└─────────────────────┘
```

## Installation

### 1. Clone or navigate to the project directory

```powershell
cd "e:\VIDEO KURSY EN\ai hallucination\ownrag"
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pnpm install  # If using pnpm for Python packages
# OR
pip install -r requirements.txt
```

### 4. Set up environment variables

Ensure your `.env` file contains:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Option 1: Using the Main Interface (Recommended)

```python
from src.rag import RAGSystem

# Initialize the RAG system
rag = RAGSystem(persist_directory="./chroma_db", top_k=3)

# Ingest a PDF (only needed once)
rag.ingest_pdf("./data/DataMind_FAQ_EN.pdf")

# Ask questions
result = rag.ask("What is DataMind?", verbose=True)

print(result['answer'])
print(f"Sources: {len(result['sources'])} chunks from pages {[s['page'] for s in result['sources']]}")
```

### Option 2: Using Individual Modules

#### Ingestion

```python
from src.ingest import PDFIngestor

# Initialize ingestor
ingestor = PDFIngestor(persist_directory="./chroma_db")

# Ingest PDF
vectorstore = ingestor.ingest_pdf(
    pdf_path="./data/DataMind_FAQ_EN.pdf",
    chunk_size=500,
    chunk_overlap=50
)
```

#### Querying

```python
from src.query import RAGQueryEngine

# Initialize query engine
engine = RAGQueryEngine(persist_directory="./chroma_db", top_k=3)

# Ask a question
result = engine.query("What is DataMind?", verbose=True)

# Get relevant chunks without generating an answer
chunks = engine.get_relevant_chunks("What is DataMind?", k=5)
```

### Option 3: Run Demo Scripts

```powershell
# Run test script
python test_rag.py

# Or import and use in your code
python -c "from src import RAGSystem; rag = RAGSystem(); print('Ready!')"
```

## Project Structure

```
ownrag/
├── .env                    # OpenAI API key
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── test_rag.py            # Test script
├── src/                   # Source code package
│   ├── __init__.py        # Package initialization
│   ├── rag.py             # Main interface module
│   ├── ingest.py          # PDF ingestion module
│   └── query.py           # Query engine module
├── data/                  # PDF documents
│   └── DataMind_FAQ_EN.pdf
└── chroma_db/             # Persistent vector database (created after ingestion)
```

## Configuration

### Chunk Settings

Adjust in `ingest.py` or when calling `ingest_pdf()`:

- **chunk_size**: Target size of each chunk (default: 500 tokens)
- **chunk_overlap**: Overlap between chunks (default: 50 tokens)

### Retrieval Settings

Adjust in `query.py` or `RAGSystem` initialization:

- **top_k**: Number of relevant chunks to retrieve (default: 3)

### Model Settings

Edit in respective modules:

- **Embeddings**: `text-embedding-3-small` (in `ingest.py` and `query.py`)
- **Chat Model**: `gpt-4o-mini` (in `query.py`)
- **Temperature**: 0 for deterministic answers (in `query.py`)

## API Reference

### RAGSystem

Main interface for the RAG system.

#### Methods

- `ingest_pdf(pdf_path, chunk_size=500, chunk_overlap=50)` - Ingest a PDF into vector store
- `ask(question, verbose=False)` - Ask a question and get an answer
- `get_relevant_chunks(question, k=None)` - Retrieve relevant chunks without generating answer
- `is_vectorstore_ready()` - Check if vector store exists

### PDFIngestor

Handles PDF ingestion pipeline.

#### Methods

- `load_pdf(pdf_path)` - Load PDF and extract text
- `chunk_documents(documents, chunk_size, chunk_overlap)` - Split into chunks
- `create_vector_store(chunks)` - Create ChromaDB vector store
- `ingest_pdf(pdf_path, chunk_size, chunk_overlap)` - Complete ingestion pipeline
- `load_existing_vectorstore()` - Load existing vector store from disk

### RAGQueryEngine

Handles question answering.

#### Methods

- `query(question, verbose=False)` - Query the RAG system
- `get_relevant_chunks(question, k=None)` - Retrieve relevant chunks only

## Cost Estimation

Based on OpenAI pricing (as of 2024):

### Embeddings (`text-embedding-3-small`)
- $0.02 per 1M tokens
- Example: 10-page PDF ≈ 5,000 tokens ≈ $0.0001

### Chat Completion (`gpt-4o-mini`)
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens
- Example query: 3 chunks (1,500 tokens) + answer (200 tokens) ≈ $0.00034

**Total for 100 queries on 10-page PDF: ~$0.04**

## Troubleshooting

### Vector store not found
Run ingestion first:
```python
rag.ingest_pdf("./data/DataMind_FAQ_EN.pdf")
```

### OpenAI API key error
Check `.env` file contains valid `OPENAI_API_KEY`

### Import errors
Ensure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

### ChromaDB persistence issues
Delete `chroma_db/` folder and re-run ingestion

## Extending the System

### Add More PDFs

```python
rag.ingest_pdf("./data/document1.pdf")
rag.ingest_pdf("./data/document2.pdf")
# All documents will be searchable together
```

### Use Different Models

Edit `query.py`:

```python
# For better quality (more expensive)
self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

# For larger embeddings (better accuracy)
self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Add Conversational Memory

Modify `query.py` to use `ConversationalRetrievalChain` instead of `RetrievalQA`

## License

MIT

## Contributing

Feel free to submit issues and enhancement requests!
