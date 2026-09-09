from fastapi.testclient import TestClient

from app.api import app, _get_app
from src.evidence import Evidence
from src.rag_result import RAGResult


class FakeStudyApp:
    def __init__(self):
        self.next_session_id = 1

    def create_session(self):
        session = type("Session", (), {"id": self.next_session_id})()
        self.next_session_id += 1
        return session

    def ask(self, question, session_id=None):
        return self._result(question)

    def explain(self, question, session_id=None):
        return self._result(question)

    def summarize(self, question, session_id=None):
        return self._result(question)

    @staticmethod
    def _result(question):
        return RAGResult(
            question=question,
            answer="Grounded answer [notes.md].",
            evidence=[Evidence("chunk-1", "Evidence text", "notes.md", "2", 0.1)],
        )


def test_health_does_not_construct_study_app():
    app.dependency_overrides[_get_app] = lambda: (_ for _ in ()).throw(AssertionError("should not initialize"))
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_structured_evidence():
    app.dependency_overrides[_get_app] = FakeStudyApp
    try:
        response = TestClient(app).post("/ask", json={"question": "What is RAG?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == 1
    assert body["sources"] == ["notes.md, page 2"]
    assert body["evidence"][0]["id"] == "E1"


def test_api_rejects_blank_or_missing_questions():
    client = TestClient(app)
    assert client.post("/ask", json={"question": " "}).status_code == 400
    assert client.post("/ask", json={}).status_code == 422
