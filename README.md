\# AI Study Agent



A local-first AI study assistant that uses RAG to answer questions from

personal study documents stored on a local machine.



\## Architecture



Local documents

→ ingestion

→ cleaning

→ chunking

→ embeddings

→ Chroma

→ semantic retrieval

→ evidence

→ cloud LLM

→ grounded answer

→ citation validation

→ study agent

→ Streamlit UI



\## Current Features



\- PDF, TXT and Markdown ingestion

\- Recursive local folder discovery

\- Text cleaning

\- Configurable chunking

\- Deterministic SHA256-based chunk IDs

\- Gemini Embedding 2

\- Persistent Chroma vector database

\- Semantic retrieval

\- Evidence-aware RAG

\- Source/page citation rendering

\- Citation validation

\- Answer mode

\- Explain mode

\- Summarize mode

\- Quiz generation

\- Flashcard generation

\- Topic comparison

\- SQLite learning memory

\- Retrieval evaluation

\- Streamlit UI



\## Technology Stack



\- Python 3.12

\- LangChain

\- Chroma

\- SQLite

\- Gemini Embedding 2

\- Gemini 3.8 Flash

\- Streamlit



\## Project Status



\### Completed



\- Document ingestion

\- Cleaning and chunking

\- Embeddings

\- Chroma indexing

\- Semantic retrieval

\- RAG pipeline

\- Citation validation

\- Study modes

\- SQLite learning memory

\- Retrieval evaluation



\### Current



\- Streamlit UI

\- Live Gemini generation validation



\### Planned



\- LangGraph orchestration

\- Proper agent state

\- Production hardening

\- Provider abstraction

\- Incremental indexing

\- Observability

\- Final end-to-end validation



\## Local / Cloud Architecture



\### Local



\- Study documents

\- Document processing

\- Chroma

\- SQLite

\- Application



\### Cloud



\- Gemini embeddings

\- Gemini generation



Original study documents remain local. Only the text required for embedding

or generation is sent to the cloud APIs.



\## Important



Create a `.env` file locally containing:



GEMINI\_API\_KEY=your\_key



Do not commit `.env`.



\## Development



Create virtual environment:



```powershell

py -3.12 -m venv venv

