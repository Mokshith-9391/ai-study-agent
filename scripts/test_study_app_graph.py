from pathlib import Path
from tempfile import TemporaryDirectory

from src.evidence import Evidence
from src.memory.database import StudyDatabase
from src.memory.repository import MemoryRepository
from src.rag_result import RAGResult
from src.study_app import StudyApp


class FakeLLM:
    """Offline LLM replacement used only for integration testing."""

    def generate(self, prompt: str) -> str:
        if "Return only the answer" in prompt:
            return "This is a grounded test answer. [E1]"

        return "This is a test response. [E1]"


class FakeRAG:
    """Minimal RAG interface required by the LangGraph nodes."""

    def __init__(self) -> None:
        self.llm = FakeLLM()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[Evidence]:
        if not question.strip():
            raise ValueError("Question must not be empty.")

        evidence = Evidence(
            chunk_id="test-chunk-001",
            content=(
                "This is deterministic test evidence used to verify "
                "LangGraph, StudyApp, and SQLite integration."
            ),
            source="test/study_notes.txt",
            page=None,
            distance=0.1,
        )

        return [evidence]

    def build_context(
        self,
        evidence: list[Evidence],
    ) -> str:
        if not evidence:
            raise ValueError("Evidence must not be empty.")

        lines = []

        for index, item in enumerate(evidence, start=1):
            lines.append(
                f"[E{index}] {item.content}"
            )

        return "\n\n".join(lines)


class FakeAgent:
    """
    Offline StudyAgent replacement.

    The graph only needs the agent's RAG interface for generation,
    while StudyApp uses the session methods.
    """

    def __init__(self) -> None:
        self.rag = FakeRAG()
        self._session_id = None

    def start_session(self):
        raise RuntimeError(
            "FakeAgent.start_session should not be used directly "
            "when StudyApp owns the injected MemoryRepository."
        )

    def end_session(self):
        self._session_id = None

    def get_all_progress(self):
        return []


def test_study_app_session_memory_integration() -> None:
    print(
        "STUDY APP ↔ LANGGRAPH ↔ SQLITE INTEGRATION TEST"
    )
    print("=" * 60)

    with TemporaryDirectory(
        prefix="study_app_memory_test_"
    ) as temp_dir:

        database_path = (
            Path(temp_dir) / "test_memory.db"
        )

        print("Creating isolated test database...")

        database = StudyDatabase(database_path)
        repository = MemoryRepository(database)

        print("  Database:", database_path)
        print("  Temporary database: PASSED")

        # ---------------------------------------------------------
        # Create an application with isolated dependencies.
        # ---------------------------------------------------------

        fake_agent = FakeAgent()

        app = StudyApp(
            agent=fake_agent,
            memory_repository=repository,
        )

        # ---------------------------------------------------------
        # Start application session.
        # ---------------------------------------------------------

        session = repository.start_session()

        app._session_id = session.id

        assert app.session_id == session.id

        print(
            "  Session created and stored: PASSED"
        )

        # ---------------------------------------------------------
        # First graph interaction.
        # ---------------------------------------------------------

        print("Running first graph interaction...")

        state_1 = app.run(
            mode="answer",
            user_request="What is RAG?",
        )

        assert state_1["result"]
        assert isinstance(state_1["result"], str)

        assert state_1["session_id"] == session.id
        assert len(state_1["evidence"]) == 1
        assert state_1["previous_messages"] == []

        print(
            "  First graph execution: PASSED"
        )

        # ---------------------------------------------------------
        # Verify first interaction persisted.
        # ---------------------------------------------------------

        messages = repository.get_messages(
            session.id
        )

        assert len(messages) == 2

        assert messages[0].role == "user"
        assert messages[0].content == "What is RAG?"

        assert messages[1].role == "assistant"
        assert messages[1].content == state_1["result"]

        print(
            "  First interaction persisted: PASSED"
        )

        # ---------------------------------------------------------
        # Second graph interaction.
        # ---------------------------------------------------------

        print("Running second graph interaction...")

        state_2 = app.run(
            mode="answer",
            user_request="Explain RAG again.",
        )

        assert state_2["result"]
        assert state_2["session_id"] == session.id

        # Previous interaction should now be available.
        assert len(state_2["previous_messages"]) == 2

        print(
            "  Second graph execution: PASSED"
        )

        # ---------------------------------------------------------
        # Verify persistent memory growth.
        # ---------------------------------------------------------

        messages = repository.get_messages(
            session.id
        )

        assert len(messages) == 4

        assert messages[2].role == "user"
        assert messages[2].content == "Explain RAG again."

        assert messages[3].role == "assistant"
        assert messages[3].content == state_2["result"]

        print(
            "  Persistent memory growth: PASSED"
        )

        # ---------------------------------------------------------
        # Verify session lifecycle.
        # ---------------------------------------------------------

        repository.end_session(session.id)

        refreshed_messages = repository.get_messages(
            session.id
        )

        assert len(refreshed_messages) == 4

        print(
            "  Session ending: PASSED"
        )

        database.close()

    print()
    print(
        "ALL STUDY APP ↔ LANGGRAPH ↔ SQLITE "
        "INTEGRATION TESTS PASSED"
    )


if __name__ == "__main__":
    test_study_app_session_memory_integration()