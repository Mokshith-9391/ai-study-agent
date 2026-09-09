# AI Study Agent

AI Study Agent is a local-first Retrieval-Augmented Generation (RAG) backend for personal study material. It indexes local PDF, TXT, and Markdown files, retrieves relevant passages, and returns Gemini-generated answers grounded in that evidence with source citations. FastAPI provides the localhost interface and automatic API documentation.

## Architecture

```text
Study documents → ingestion → conservative cleaning → chunking
→ Gemini embeddings → persistent Chroma → retrieval → evidence
→ LangGraph → Gemini grounded generation → cited answer → SQLite memory
```

Documents, Chroma data, and SQLite memory stay local. Only text needed for embedding or a grounded answer is sent to Gemini.

## Requirements

- Windows with Python 3.12 recommended
- A Gemini API key
- Internet access for Gemini calls and package installation

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

Set `GEMINI_API_KEY` in `.env`. Do not put a key in source code or commit `.env`.

Optional path settings in `.env`:

```dotenv
STUDY_FOLDER=data/study
VECTORSTORE_PATH=vectorstore
DATABASE_PATH=data/study_agent.db
```

## Index study material

Place PDF, TXT, or Markdown files anywhere beneath `data/study/`, then run:

```powershell
python -m scripts.index_study
```

Chunk IDs are deterministic and indexing uses Chroma upserts, so unchanged chunks are not duplicated.

## Run the localhost API

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI or [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json) for the schema.

## API endpoints

- `GET /health` — safe health check; never calls Gemini.
- `POST /session/start` — create a durable SQLite conversation session.
- `POST /ask` — answer a question from indexed evidence.
- `POST /explain` — explain a topic from indexed evidence.
- `POST /summarize` — summarize a topic from indexed evidence.

PowerShell example:

```powershell
$body = @{ question = "What is Retrieval-Augmented Generation?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body $body
```

Pass the returned `session_id` in later requests to preserve conversation history:

```powershell
$body = @{ question = "Explain it more simply"; session_id = 1 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/explain -ContentType "application/json" -Body $body
```

## Testing

```powershell
python -m compileall src app scripts
python -m pytest
```

The tests use fakes for Gemini and cover the API routing and validation without consuming API quota. Existing script-level checks remain available under `scripts/`.

## Project structure

```text
app/            FastAPI localhost API (legacy Streamlit file is retained)
src/            ingestion, RAG, LangGraph orchestration, and SQLite memory
scripts/        indexing and standalone verification scripts
tests/          pytest API tests
data/study/     local study material (ignored by Git)
vectorstore/   local Chroma persistence (ignored by Git)
```

## Design decisions

- **Chroma** provides persistent local vector search.
- **Gemini embeddings and generation** provide managed semantic retrieval and grounded responses.
- **LangGraph** keeps retrieval, generation, validation, and memory flow explicit.
- **SQLite** gives a durable, lightweight single-user session store.
- **FastAPI** supplies a clean localhost API, OpenAPI schema, and Swagger UI without a frontend.

## Current limitations

- This is a single-user localhost application; it has no authentication or cloud deployment setup.
- Retrieval is dense-vector baseline retrieval; hybrid retrieval and reranking are intentionally not implemented.
- Quiz and flashcard generation remain available in the core application but have no API/UI focus in this MVP.
- A valid Gemini API key and indexed material are required for answer-generating endpoints.
