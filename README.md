# RAG System

Ask questions about your documents. The system reads PDFs, stores them in a database, and answers questions using GPT-5 nano.

## What It Does

1. Reads PDF files
2. Breaks text into small chunks
3. Stores chunks in ChromaDB
4. Finds relevant chunks for your question
5. Generates answers using GPT-5 nano

## Install

```powershell

pip install -r requirements.txt
```

Create `.env` file from template:
```powershell
Copy-Item .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-key-here
```

## Use

### Quick Start

```python
from src.rag import RAGSystem

rag = RAGSystem()
rag.ingest_folder("./data")  # Load all files once
result = rag.ask("What is DataMind?")
print(result['answer'])
```

### Run Test

```powershell
python test_rag.py
```

## Features

**Multi-format support**: PDF, TXT, MD files

**Smart tracking**: Skips files you already loaded

**Folder ingestion**: Load all files at once
```python
rag.ingest_folder("./data")
```

**Single file**: Load one file
```python
rag.ingest_file("./data/document.pdf")
```

**Force reload**: Re-process files
```python
rag.ingest_folder("./data", force=True)
```

**Check loaded files**:
```python
files = rag.get_ingested_files()
for path in files:
    print(path)
```

## Structure

```
ownrag/
├── .env                 # API key
├── requirements.txt     # Dependencies
├── test_rag.py         # Test script
├── src/                # Code
│   ├── ingest.py       # Load documents
│   ├── query.py        # Answer questions
│   └── rag.py          # Main interface
├── data/               # Your documents
└── chroma_db/          # Database (auto-created)
```

## Settings

**Chunk size**: How much text per chunk (default: 500)
```python
rag.ingest_file("doc.pdf", chunk_size=1000)
```

**Chunk overlap**: Text shared between chunks (default: 50)
```python
rag.ingest_file("doc.pdf", chunk_overlap=100)
```

**Results per query (top_k)**: How many chunks to use (default: 3)

`top_k=3` means the system retrieves the 3 most relevant chunks to answer your question.

**How it works:**
1. You ask a question
2. The retriever finds all possible matches
3. `top_k` decides how many of the best ones go to the model

**Why 3?**
- 1-2 chunks might miss context
- 5-10 chunks might overload with noise
- 3 gives enough info without bloating the prompt

**When to adjust:**

*Chunk size:*
- Small chunks → raise to 5-10
- Large chunks → lower to 1-3

*Question type:*
- Simple question → smaller top_k
- Broad question → larger top_k

*Model limits:*
- Small token window → lower top_k to avoid overflow

```python
# Simple questions
rag = RAGSystem(top_k=2)

# Complex questions needing more context
rag = RAGSystem(top_k=7)
```

## Cost

Based on OpenAI prices:

**Load 10-page PDF**: ~$0.0001
**Ask 100 questions**: ~$0.04

Total: About $0.04 for 100 questions on a 10-page document.

## Fix Problems

**"Vector store not found"**
```python
rag.ingest_folder("./data")
```

**"API key error"**
Check your `.env` file has `OPENAI_API_KEY=...`

**Import errors**
```powershell
pip install -r requirements.txt
```

**Database issues**
Delete `chroma_db/` folder and reload files

## Advanced

### Load Many Files

```python
rag.ingest_folder("./data")
# All files searchable together
```

### Use Better Models

Edit `src/query.py`:
```python
# Better answers (costs more)
self.llm = ChatOpenAI(model="gpt-4o")

# Better search (costs more)
self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Get Chunks Without Answers

```python
chunks = rag.get_relevant_chunks("What is DataMind?", k=5)
for chunk in chunks:
    print(chunk['content'])
```

### Use Individual Modules

```python
from src.ingest import DocumentIngestor
from src.query import RAGQueryEngine

# Load files
ingestor = DocumentIngestor()
ingestor.ingest_folder("./data")

# Ask questions
engine = RAGQueryEngine()
result = engine.query("What is DataMind?")
```

## How It Works

```
PDF → Load text → Split into chunks → Create embeddings → Store in ChromaDB

Question → Create embedding → Find similar chunks → Send to GPT-5 nano → Get answer
```

The system finds text similar to your question and sends it to GPT-5 nano for an answer. Each answer includes page numbers so you can check the source.

## License

MIT
