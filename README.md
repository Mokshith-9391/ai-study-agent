# AI Study Agent

A local-first Retrieval-Augmented Generation (RAG) study assistant that indexes PDF, TXT, and Markdown documents, retrieves relevant passages using Gemini embeddings and Chroma vector search, and generates source-grounded answers through a LangGraph workflow with persistent SQLite memory. Streamlit provides the primary interactive interface; FastAPI is available as an optional local API.

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit / FastAPI                        │
├─────────────────────────────────────────────────────────────────────┤
│                            StudyApp                                │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │                    LangGraph Workflow                      │    │
│   │                                                           │    │
│   │   START → load_memory → route_request → retrieve_evidence │    │
│   │         → build_context → generate_* → validate_result    │    │
│   │         → save_interaction → END                          │    │
│   └───────────────────────────────────────────────────────────┘    │
│         ↕                    ↕                    ↕                 │
│   StudyAgent          RAGPipeline          MemoryRepository        │
│         ↕                    ↕                    ↕                 │
│   Mode Prompts     GeminiEmbedder + LLM     SQLite Database       │
│                    ChromaVectorStore                               │
└─────────────────────────────────────────────────────────────────────┘
```

All study documents, vector embeddings, and conversation history remain on your local machine. Only the text required for embedding or answer generation is sent to the Gemini API.

## RAG Flow

1. **Ingestion** — PDF, TXT, and Markdown files are loaded from a local study folder, cleaned with conservative boilerplate removal, and split into overlapping chunks using LangChain text splitters.
2. **Embedding** — Each chunk is embedded using the Gemini Embedding API (`gemini-embedding-2`, 768 dimensions).
3. **Indexing** — Chunk embeddings are upserted into a persistent Chroma vector store with deterministic IDs, preventing duplicate entries on re-indexing.
4. **Retrieval** — A user query is embedded and matched against stored chunks using Chroma's cosine-distance search (default top-5).
5. **Context Assembly** — Retrieved evidence chunks are formatted with labels (`[E1]`, `[E2]`, …) and assembled into an LLM-ready context block.
6. **Generation** — Gemini generates a grounded response using a mode-specific prompt (answer, explain, summarize, quiz, flashcards, or compare) and strict citation rules.
7. **Validation** — The generated output is validated for structural correctness (JSON schema for structured modes, citation integrity for text modes).
8. **Memory** — The interaction is persisted to SQLite: session state, user/assistant messages, quiz attempts, flashcard reviews, and per-topic learning progress.

## Key Components

| Component | Technology | Purpose |
|---|---|---|
| Embeddings | Gemini Embedding API | Semantic vector representations of text chunks |
| Vector Store | Chroma | Persistent local similarity search with cosine distance |
| Orchestration | LangGraph | Explicit DAG workflow connecting retrieval → generation → validation → memory |
| Generation | Gemini API | Grounded, citation-aware answer generation |
| Memory | SQLite | Durable sessions, messages, quiz attempts, flashcard reviews, and topic progress |
| Ingestion | LangChain loaders + custom pipeline | PDF/TXT/MD loading, cleaning, and recursive chunking |
| UI | Streamlit | Interactive study interface with session management |
| API | FastAPI | Optional local REST API with OpenAPI/Swagger documentation |

## Study Modes

- **Ask** — Answer a question grounded in indexed evidence with citations.
- **Explain** — Generate a structured explanation of a topic from evidence.
- **Summarize** — Produce a concise study summary from evidence.
- **Quiz** — Generate validated multiple-choice questions with explanations.
- **Flashcards** — Create study flashcards grounded in evidence.
- **Compare** — Compare two topics using evidence-backed analysis.
- **Progress** — View quiz accuracy and flashcard retention per topic.

## Requirements

- Python 3.12+ (recommended on Windows)
- A [Gemini API key](https://aistudio.google.com/apikey)
- Internet access for Gemini API calls and package installation

## Setup

```powershell
git clone https://github.com/Mokshith-9391/ai-study-agent.git
cd ai-study-agent

py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
```

Edit `.env` and set your Gemini API key:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

> **Do not** commit `.env` or put API keys in source code. The `.gitignore` already excludes `.env` files.

Optional configuration in `.env`:

```dotenv
STUDY_FOLDER=data/study           # Where study documents are stored
VECTORSTORE_PATH=vectorstore      # Where Chroma persists embeddings
DATABASE_PATH=data/study_agent.db # Where SQLite stores session data
```

## Indexing Study Material

Place PDF, TXT, or Markdown files anywhere beneath `data/study/`, then run:

```powershell
python -m scripts.index_study
```

Chunk IDs are deterministic (based on content hash), and indexing uses Chroma upserts. Re-running the script on unchanged files does not create duplicates.

## Running the Streamlit App

```powershell
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. It is configured to listen on `0.0.0.0`, so other devices on the same network can access it.

> **LAN access:** Run `ipconfig` in PowerShell and find the IPv4 address under your Wi-Fi adapter (e.g. `192.168.x.x`). Other devices can then open `http://<your-ipv4>:8501`. If Windows Firewall prompts, allow Python on private networks.

## Optional FastAPI API

```powershell
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for Swagger UI; the OpenAPI schema is at `/openapi.json`.

> **LAN access:** Use the same IPv4 address approach described above, substituting port `8000`.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Dependency-free health check; never calls Gemini |
| `GET` | `/` | Redirects to Swagger UI |
| `POST` | `/session/start` | Create a durable SQLite conversation session |
| `POST` | `/ask` | Answer a question from indexed evidence |
| `POST` | `/explain` | Explain a topic from indexed evidence |
| `POST` | `/summarize` | Summarize a topic from indexed evidence |

**Example usage:**

```powershell
# Ask a question
$body = @{ question = "What is Retrieval-Augmented Generation?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body $body

# Continue a conversation with session memory
$body = @{ question = "Explain it more simply"; session_id = 1 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/explain -ContentType "application/json" -Body $body
```

## Testing

```powershell
python -m compileall src app scripts
python -m pytest
```

Tests use fakes for all Gemini and vector store dependencies, so they run offline without consuming API quota. The test suite covers API routing, request validation, and response structure.

## Project Structure

```text
ai-study-agent/
├── app/
│   ├── api.py               # FastAPI REST API with Swagger docs
│   └── streamlit_app.py     # Streamlit interactive study interface
├── src/
│   ├── config.py            # Central settings from environment variables
│   ├── embedding.py         # Gemini Embedding API client
│   ├── vectorstore.py       # Chroma persistent vector store
│   ├── llm.py               # Gemini generation client
│   ├── rag.py               # RAG pipeline: retrieve → context → generate
│   ├── rag_result.py        # Structured RAG result model
│   ├── evidence.py          # Retrieved evidence dataclass
│   ├── citations.py         # Citation rendering for grounded answers
│   ├── context.py           # Evidence-to-context formatting
│   ├── validation.py        # Output validation for structured modes
│   ├── study_agent.py       # High-level study agent with all modes
│   ├── study_app.py         # Application layer: sessions + LangGraph
│   ├── study_models.py      # Pydantic models for quiz/flashcard/compare
│   ├── graph/
│   │   ├── workflow.py      # LangGraph DAG definition
│   │   ├── nodes.py         # Workflow node implementations
│   │   └── state.py         # LangGraph state schema
│   ├── ingestion/
│   │   ├── loader.py        # File discovery and document loading
│   │   ├── cleaner.py       # Conservative text cleaning
│   │   ├── chunker.py       # Recursive text chunking
│   │   └── boilerplate.py   # Boilerplate detection and removal
│   ├── memory/
│   │   ├── database.py      # SQLite schema and connection management
│   │   ├── models.py        # Session, message, and progress models
│   │   └── repository.py    # Data access layer for memory operations
│   └── evaluation/
│       ├── metrics.py       # Retrieval quality metrics
│       └── retrieval_eval.py# Retrieval evaluation utilities
├── scripts/
│   ├── index_study.py       # CLI: index study documents into Chroma
│   └── *.py                 # Standalone verification and test scripts
├── tests/
│   └── test_api.py          # Pytest suite for FastAPI endpoints
├── .streamlit/
│   └── config.toml          # Streamlit server configuration
├── .env.example             # Template for environment variables
├── .gitignore               # Excludes secrets, data, and generated files
├── requirements.txt         # Python dependencies with version bounds
└── pyproject.toml           # Pytest configuration
```

### Data directories (git-ignored)

| Directory | Contents |
|---|---|
| `data/study/` | Local study documents (PDF, TXT, MD) |
| `vectorstore/` | Chroma persistence directory |
| `data/study_agent.db` | SQLite session and progress database |

## Limitations

- **Single-user localhost application** — no authentication, multi-tenancy, or cloud deployment.
- **Dense-vector retrieval only** — no hybrid search or reranking; relies on Gemini embedding quality.
- **Gemini API dependency** — a valid API key and internet access are required for embedding and generation.
- **No streaming** — responses are returned after full generation completes.

## Future Improvements

- Hybrid retrieval combining dense vectors with BM25 keyword search
- Reranking retrieved passages before generation
- Streaming responses in the Streamlit UI
- Support for additional document formats (DOCX, EPUB, HTML)
- Batch embedding for faster ingestion of large document collections
- Export study progress and generated materials

## License

This project is provided for educational and portfolio purposes.
