from dataclasses import dataclass
import json
import tempfile
from pathlib import Path

from src.evidence import Evidence
from src.graph.workflow import build_study_graph
from src.memory.database import StudyDatabase
from src.memory.repository import MemoryRepository


# ============================================================
# FAKE RAG
# ============================================================


@dataclass
class FakeRAG:

    class FakeLLM:

        def generate(
            self,
            prompt: str,
        ) -> str:

            return (
                "Retrieval-Augmented Generation retrieves "
                "relevant information before generation. "
                "[E1]"
            )

    def __init__(self):
        self.llm = self.FakeLLM()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ):

        return [
            Evidence(
                chunk_id="memory-test-001",
                content=(
                    "Retrieval-Augmented Generation retrieves "
                    "relevant information before generation."
                ),
                source="memory_test.txt",
                page=None,
                distance=0.1,
            )
        ]

    def build_context(
        self,
        evidence,
    ):

        return "\n".join(
            f"[E{i}] {item.content}"
            for i, item in enumerate(
                evidence,
                start=1,
            )
        )


# ============================================================
# FAKE AGENT
# ============================================================


class FakeAgent:

    def __init__(self):
        self.rag = FakeRAG()

    def _parse_json(
        self,
        response,
    ):
        return json.loads(response)


# ============================================================
# MAIN TEST
# ============================================================


def main() -> None:

    print(
        "LANGGRAPH ↔ SQLITE MEMORY TEST"
    )
    print("=" * 55)

    # --------------------------------------------------------
    # Temporary database
    # --------------------------------------------------------

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="study_agent_memory_test_"
        )
    )

    database_path = (
        temporary_directory
        / "test_memory.db"
    )

    print(
        "\nCreating isolated test database..."
    )

    database = StudyDatabase(
        database_path
    )

    repository = MemoryRepository(
        database=database
    )

    print(
        f"  Database: {database_path}"
    )

    print(
        "  Temporary database: PASSED"
    )

    try:

        # ----------------------------------------------------
        # Session
        # ----------------------------------------------------

        print(
            "\nCreating study session..."
        )

        session = repository.start_session()

        assert session is not None

        assert session.id is not None

        session_id = session.id

        print(
            f"  Session ID: {session_id}"
        )

        print(
            "  Session creation: PASSED"
        )

        # ----------------------------------------------------
        # Graph
        # ----------------------------------------------------

        print(
            "\nBuilding LangGraph..."
        )

        graph = build_study_graph(
            agent=FakeAgent(),
            memory_repository=repository,
        )

        assert graph is not None

        print(
            "  Graph creation: PASSED"
        )

        # ----------------------------------------------------
        # First execution
        # ----------------------------------------------------

        print(
            "\nRunning first graph execution..."
        )

        first_result = graph.invoke(
            {
                "mode": "answer",
                "user_request": "What is RAG?",
                "session_id": session_id,
            }
        )

        assert first_result is not None

        assert first_result["result"]

        print(
            "  First graph execution: PASSED"
        )

        # ----------------------------------------------------
        # Verify first interaction
        # ----------------------------------------------------

        print(
            "\nChecking first interaction..."
        )

        messages = repository.get_messages(
            session_id
        )

        assert len(messages) == 2, (
            f"Expected 2 messages after first run, "
            f"got {len(messages)}."
        )

        assert messages[0].role == "user"

        assert messages[0].content == "What is RAG?"

        assert messages[1].role == "assistant"

        assert (
            "Retrieval-Augmented Generation"
            in messages[1].content
        )

        print(
            "  User message persisted: PASSED"
        )

        print(
            "  Assistant message persisted: PASSED"
        )

        # ----------------------------------------------------
        # Memory loading
        # ----------------------------------------------------

        print(
            "\nChecking memory loading..."
        )

        second_result = graph.invoke(
            {
                "mode": "answer",
                "user_request": "Explain RAG.",
                "session_id": session_id,
            }
        )

        assert second_result is not None

        assert (
            "previous_messages"
            in second_result
        )

        previous_messages = (
            second_result[
                "previous_messages"
            ]
        )

        assert len(previous_messages) == 2, (
            "The second execution should load "
            "the two messages from the first execution."
        )

        print(
            "  Previous messages loaded: PASSED"
        )

        # ----------------------------------------------------
        # Second interaction
        # ----------------------------------------------------

        print(
            "\nChecking second interaction..."
        )

        messages_after_second_run = (
            repository.get_messages(
                session_id
            )
        )

        assert len(
            messages_after_second_run
        ) == 4, (
            f"Expected 4 messages after second run, "
            f"got {len(messages_after_second_run)}."
        )

        assert (
            messages_after_second_run[2].role
            == "user"
        )

        assert (
            messages_after_second_run[2].content
            == "Explain RAG."
        )

        assert (
            messages_after_second_run[3].role
            == "assistant"
        )

        print(
            "  Second user message persisted: PASSED"
        )

        print(
            "  Second assistant message persisted: PASSED"
        )

        # ----------------------------------------------------
        # Persistent growth
        # ----------------------------------------------------

        print(
            "\nChecking persistent memory growth..."
        )

        assert (
            len(messages_after_second_run)
            > len(messages)
        )

        print(
            "  Persistent memory growth: PASSED"
        )

        # ----------------------------------------------------
        # End session
        # ----------------------------------------------------

        print(
            "\nEnding study session..."
        )

        repository.end_session(
            session_id
        )

        print(
            "  Session ending: PASSED"
        )

        print()
        print(
            "ALL LANGGRAPH ↔ SQLITE MEMORY TESTS PASSED"
        )

    finally:

        database.connection.close()

        print(
            "\nDatabase connection closed."
        )


if __name__ == "__main__":
    main()