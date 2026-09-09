"""FastAPI localhost interface for the AI Study Agent."""

from contextlib import asynccontextmanager
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.rag_result import RAGResult
from src.study_app import StudyApp


logger = logging.getLogger(__name__)


class StudyRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    session_id: int | None = Field(default=None, gt=0)


class SessionResponse(BaseModel):
    session_id: int


class EvidenceResponse(BaseModel):
    id: str
    source: str
    page: str | None
    chunk_id: str
    distance: float
    content: str


class StudyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question: str
    answer: str
    session_id: int
    sources: list[str]
    evidence: list[EvidenceResponse]


def create_study_app() -> StudyApp:
    """Factory kept separate so tests can replace the dependency."""
    return StudyApp()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logger.info("AI Study Agent API started")
    yield
    logger.info("AI Study Agent API stopped")


app = FastAPI(
    title="AI Study Agent API",
    version="1.0.0",
    description="Local RAG study assistant with cited Gemini answers.",
    lifespan=lifespan,
)


def _get_app() -> StudyApp:
    # Lazy initialization keeps /health independent of Gemini credentials.
    instance = getattr(app.state, "study_app", None)
    if instance is None:
        instance = create_study_app()
        app.state.study_app = instance
    return instance


StudyAppDependency = Annotated[StudyApp, Depends(_get_app)]


def _response(result: RAGResult, session_id: int) -> StudyResponse:
    return StudyResponse(
        question=result.question,
        answer=result.answer,
        session_id=session_id,
        sources=result.sources,
        evidence=[
            EvidenceResponse(
                id=f"E{index}",
                source=item.source,
                page=item.page,
                chunk_id=item.chunk_id,
                distance=float(item.distance),
                content=item.content,
            )
            for index, item in enumerate(result.evidence, start=1)
        ],
    )


def _run_request(request: StudyRequest, mode: str, study_app: StudyApp) -> StudyResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="question must not be blank")

    session_id = request.session_id
    if session_id is None:
        session_id = study_app.create_session().id

    try:
        if mode == "ask":
            result = study_app.ask(question, session_id=session_id)
        elif mode == "explain":
            result = study_app.explain(question, session_id=session_id)
        else:
            result = study_app.summarize(question, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("Study request failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:  # Keep provider details and stack traces server-side.
        logger.exception("Unexpected study request failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="The study request could not be completed.") from exc

    return _response(result, session_id)


@app.get("/health")
def health() -> dict[str, str]:
    """Fast, dependency-safe health check; it never calls Gemini."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send browser visitors to the interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/session/start", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(study_app: StudyAppDependency) -> SessionResponse:
    return SessionResponse(session_id=study_app.create_session().id)


@app.post("/ask", response_model=StudyResponse)
def ask(request: StudyRequest, study_app: StudyAppDependency) -> StudyResponse:
    return _run_request(request, "ask", study_app)


@app.post("/explain", response_model=StudyResponse)
def explain(request: StudyRequest, study_app: StudyAppDependency) -> StudyResponse:
    return _run_request(request, "explain", study_app)


@app.post("/summarize", response_model=StudyResponse)
def summarize(request: StudyRequest, study_app: StudyAppDependency) -> StudyResponse:
    return _run_request(request, "summarize", study_app)
