from dataclasses import dataclass
import json

from src.evidence import Evidence
from src.rag_result import RAGResult
from src.study_app import StudyApp


# ============================================================
# FAKE RAG
# ============================================================


@dataclass
class FakeRAG:
    class FakeLLM:
        def generate(self, prompt: str) -> str:

            # ------------------------------
            # Quiz
            # ------------------------------

            if "multiple-choice quiz" in prompt:
                return """
[
  {
    "question": "What does RAG do?",
    "options": [
      "Retrieves relevant information",
      "Deletes documents",
      "Compiles Python",
      "Creates a database"
    ],
    "answer": "Retrieves relevant information",
    "explanation": "RAG retrieves relevant information before generation.",
    "evidence": ["E1"]
  }
]
"""

            # ------------------------------
            # Flashcards
            # ------------------------------

            if "flashcards" in prompt:
                return """
[
  {
    "front": "What is RAG?",
    "back": "Retrieval-Augmented Generation retrieves relevant information before generation.",
    "evidence": ["E1"]
  }
]
"""

            # ------------------------------
            # Comparison
            # ------------------------------

            if "Compare these two topics" in prompt:
                return """
{
  "topic_a": "RAG",
  "topic_b": "Fine-tuning",
  "comparison": "RAG retrieves external information while fine-tuning changes model parameters.",
  "evidence": ["E1"]
}
"""

            # ------------------------------
            # Answer / Explain / Summary
            # ------------------------------

            return (
                "Retrieval-Augmented Generation retrieves "
                "relevant information before generating "
                "an answer. [E1]"
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
                chunk_id="streamlit-001",
                content=(
                    "Retrieval-Augmented Generation retrieves "
                    "relevant information before generating "
                    "an answer."
                ),
                source="streamlit_test.txt",
                page=None,
                distance=0.1,
            )
        ]

    def build_context(self, evidence):
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

    def _parse_json(self, response):
        return json.loads(response)

    def start_session(self):
        return type("Session", (), {"id": 1})()

    def end_session(self):
        pass

    def get_all_progress(self):
        return []


# ============================================================
# TEST HELPERS
# ============================================================


def check_string_result(
    name: str,
    result,
):
    assert isinstance(result, RAGResult), (
        f"{name} must return a RAGResult."
    )

    assert result.answer.strip(), (
        f"{name} returned an empty string."
    )

    print(
        f"  {name}: PASSED"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "STREAMLIT BACKEND / LANGGRAPH TEST"
    )
    print("=" * 50)

    app = StudyApp(
        agent=FakeAgent()
    )

    # --------------------------------------------------------
    # Graph existence
    # --------------------------------------------------------

    print(
        "\nTesting LangGraph connection..."
    )

    assert app.graph is not None

    print(
        "  StudyApp → LangGraph: PASSED"
    )

    # --------------------------------------------------------
    # Ask
    # --------------------------------------------------------

    print(
        "\nTesting Ask..."
    )

    result = app.ask(
        "What is RAG?"
    )

    check_string_result(
        "Ask",
        result,
    )

    # --------------------------------------------------------
    # Explain
    # --------------------------------------------------------

    print(
        "\nTesting Explain..."
    )

    result = app.explain(
        "RAG"
    )

    check_string_result(
        "Explain",
        result,
    )

    # --------------------------------------------------------
    # Summarize
    # --------------------------------------------------------

    print(
        "\nTesting Summarize..."
    )

    result = app.summarize(
        "RAG"
    )

    check_string_result(
        "Summarize",
        result,
    )

    # --------------------------------------------------------
    # Quiz
    # --------------------------------------------------------

    print(
        "\nTesting Quiz..."
    )

    result = app.quiz(
        "RAG",
        number_of_questions=1,
    )

    assert isinstance(result, list)
    assert len(result) == 1

    print(
        "  Quiz: PASSED"
    )

    # --------------------------------------------------------
    # Flashcards
    # --------------------------------------------------------

    print(
        "\nTesting Flashcards..."
    )

    result = app.flashcards(
        "RAG",
        number_of_cards=1,
    )

    assert isinstance(result, list)
    assert len(result) == 1

    print(
        "  Flashcards: PASSED"
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    print(
        "\nTesting Compare..."
    )

    result = app.compare(
        "RAG",
        "Fine-tuning",
    )

    assert result is not None

    print(
        "  Compare: PASSED"
    )

    # --------------------------------------------------------
    # Session API
    # --------------------------------------------------------

    print(
        "\nTesting Session API..."
    )

    session = app.start_session()

    assert session.id == 1

    app.end_session()

    print(
        "  Session API: PASSED"
    )

    # --------------------------------------------------------
    # Progress API
    # --------------------------------------------------------

    print(
        "\nTesting Progress API..."
    )

    progress = app.progress()

    assert isinstance(progress, list)

    print(
        "  Progress API: PASSED"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "ALL STREAMLIT BACKEND TESTS PASSED"
    )


if __name__ == "__main__":
    main()
