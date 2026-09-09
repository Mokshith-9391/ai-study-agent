from dataclasses import dataclass
import json

from src.evidence import Evidence
from src.graph.workflow import build_study_graph


# ============================================================
# FAKE RAG / AGENT
# ============================================================


@dataclass
class FakeRAG:
    class FakeLLM:
        def __init__(self, response: str):
            self.response = response

        def generate(self, prompt: str) -> str:
            return self.response

    def __init__(self, response: str):
        self.llm = self.FakeLLM(response)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ):
        return [
            Evidence(
                chunk_id="offline-001",
                content=(
                    "Retrieval-Augmented Generation retrieves relevant "
                    "information before generating an answer."
                ),
                source="offline_test.txt",
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


class FakeAgent:
    def __init__(self, response: str):
        self.rag = FakeRAG(response)

    def _parse_json(self, response):
        return json.loads(response)


# ============================================================
# FAILURE ASSERTION HELPER
# ============================================================


def expect_failure(
    graph,
    state: dict,
    expected_text: str | None = None,
    expected_exception: type[Exception] | None = None,
) -> None:
    try:
        graph.invoke(state)

    except Exception as exc:
        message = str(exc)

        if expected_exception is not None:
            assert isinstance(exc, expected_exception), (
                f"Expected exception type "
                f"{expected_exception.__name__}, "
                f"got {type(exc).__name__}: "
                f"{message}"
            )

        if expected_text is not None:
            assert expected_text.lower() in message.lower(), (
                f"Expected error containing "
                f"{expected_text!r}, "
                f"got: {message!r}"
            )

        if expected_exception is not None:
            print(
                "  REJECTED correctly: "
                f"{expected_exception.__name__}"
            )
        else:
            print(
                "  REJECTED correctly: "
                f"{expected_text}"
            )

        return

    raise AssertionError(
        "Expected graph to reject invalid output, "
        "but it succeeded."
    )


# ============================================================
# QUIZ VALIDATION TESTS
# ============================================================


def test_invalid_quiz_options():
    """
    The quiz must contain exactly four options.
    """

    response = """
[
  {
    "question": "What does RAG do?",
    "options": [
      "Retrieves information",
      "Deletes information",
      "Compiles code"
    ],
    "answer": "Retrieves information",
    "explanation": "RAG retrieves information.",
    "evidence": ["E1"]
  }
]
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "quiz",
            "user_request": "RAG",
            "number_of_questions": 1,
        },
        expected_text="exactly four items",
    )


def test_invalid_quiz_answer():
    """
    The correct answer must match one of the options.
    """

    response = """
[
  {
    "question": "What does RAG do?",
    "options": [
      "Retrieves information",
      "Deletes information",
      "Compiles code",
      "Creates a database"
    ],
    "answer": "Something else",
    "explanation": "Invalid answer.",
    "evidence": ["E1"]
  }
]
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "quiz",
            "user_request": "RAG",
            "number_of_questions": 1,
        },
        expected_text="answer",
    )


def test_invalid_quiz_evidence_reference():
    """
    E99 does not exist because the fake retrieval returns only E1.
    """

    response = """
[
  {
    "question": "What does RAG do?",
    "options": [
      "Retrieves information",
      "Deletes information",
      "Compiles code",
      "Creates a database"
    ],
    "answer": "Retrieves information",
    "explanation": "RAG retrieves information.",
    "evidence": ["E99"]
  }
]
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "quiz",
            "user_request": "RAG",
            "number_of_questions": 1,
        },
        expected_text="out of range",
    )


def test_malformed_quiz_json():
    """
    Malformed JSON must be rejected by JSON parsing.
    """

    response = """
[
  {
    "question": "What does RAG do?",
    "options":
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "quiz",
            "user_request": "RAG",
            "number_of_questions": 1,
        },
        expected_exception=json.JSONDecodeError,
    )


# ============================================================
# FLASHCARD VALIDATION TESTS
# ============================================================


def test_invalid_flashcard_evidence():
    """
    E99 does not exist because the fake retrieval returns only E1.
    """

    response = """
[
  {
    "front": "What is RAG?",
    "back": "Retrieval-Augmented Generation retrieves relevant information.",
    "evidence": ["E99"]
  }
]
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "flashcards",
            "user_request": "RAG",
            "number_of_cards": 1,
        },
        expected_text="out of range",
    )


def test_malformed_flashcard_json():
    """
    Malformed flashcard JSON must be rejected.
    """

    response = """
[
  {
    "front": "What is RAG?",
    "back":
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "flashcards",
            "user_request": "RAG",
            "number_of_cards": 1,
        },
        expected_exception=json.JSONDecodeError,
    )


# ============================================================
# COMPARISON VALIDATION TESTS
# ============================================================


def test_invalid_comparison_evidence():
    """
    E99 does not exist because the fake retrieval returns only E1.

    This intentionally tests evidence resolution rather than assuming
    that validate_comparison() rejects an empty topic field.
    """

    response = """
{
  "topic_a": "RAG",
  "topic_b": "Fine-tuning",
  "comparison": "RAG retrieves external information while fine-tuning changes model parameters.",
  "evidence": ["E99"]
}
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "compare",
            "user_request": "Compare RAG and fine-tuning.",
            "topic_a": "RAG",
            "topic_b": "Fine-tuning",
        },
        expected_text="out of range",
    )


def test_malformed_comparison_json():
    """
    Malformed comparison JSON must be rejected.
    """

    response = """
{
  "topic_a": "RAG",
  "topic_b":
"""

    graph = build_study_graph(
        FakeAgent(response)
    )

    expect_failure(
        graph,
        {
            "mode": "compare",
            "user_request": "Compare RAG and fine-tuning.",
            "topic_a": "RAG",
            "topic_b": "Fine-tuning",
        },
        expected_exception=json.JSONDecodeError,
    )


# ============================================================
# ROUTING VALIDATION
# ============================================================


def test_invalid_mode():
    """
    Unsupported study modes must be rejected before retrieval.
    """

    graph = build_study_graph(
        FakeAgent(
            "This response should never be generated."
        )
    )

    expect_failure(
        graph,
        {
            "mode": "invalid_mode",
            "user_request": "Test",
        },
        expected_text="unsupported study mode",
    )


# ============================================================
# MAIN TEST RUNNER
# ============================================================


def main() -> None:
    print(
        "LANGGRAPH VALIDATION / FAILURE TEST"
    )
    print("=" * 45)

    tests = [
        (
            "Invalid quiz options",
            test_invalid_quiz_options,
        ),
        (
            "Invalid quiz answer",
            test_invalid_quiz_answer,
        ),
        (
            "Invalid quiz evidence",
            test_invalid_quiz_evidence_reference,
        ),
        (
            "Malformed quiz JSON",
            test_malformed_quiz_json,
        ),
        (
            "Invalid flashcard evidence",
            test_invalid_flashcard_evidence,
        ),
        (
            "Malformed flashcard JSON",
            test_malformed_flashcard_json,
        ),
        (
            "Invalid comparison evidence",
            test_invalid_comparison_evidence,
        ),
        (
            "Malformed comparison JSON",
            test_malformed_comparison_json,
        ),
        (
            "Invalid mode",
            test_invalid_mode,
        ),
    ]

    for name, test in tests:
        test()

    print()
    print("ALL FAILURE CASES PASSED")
    print("LANGGRAPH VALIDATION TEST PASSED")


if __name__ == "__main__":
    main()