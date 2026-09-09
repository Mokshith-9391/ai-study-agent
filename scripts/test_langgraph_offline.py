from dataclasses import dataclass

from src.evidence import Evidence
from src.graph.workflow import build_study_graph


@dataclass
class FakeRAG:
    class FakeLLM:
        def generate(self, prompt: str) -> str:
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

            if "Compare these two topics" in prompt:
                return """
{
  "topic_a": "RAG",
  "topic_b": "Fine-tuning",
  "comparison": "RAG retrieves external information while fine-tuning changes model parameters.",
  "evidence": ["E1"]
}
"""

            return (
                "Retrieval-Augmented Generation retrieves relevant "
                "information before generating an answer. [E1]"
            )

    llm = FakeLLM()

    def retrieve(self, question: str, top_k: int = 5):
        print(
            f"  RETRIEVE: question={question!r}, top_k={top_k}"
        )

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
            for i, item in enumerate(evidence, start=1)
        )


class FakeAgent:
    def __init__(self):
        self.rag = FakeRAG()

    def _parse_json(self, response):
        import json

        return json.loads(response)


def test_mode(graph, mode, request, extra_state=None):
    state = {
        "mode": mode,
        "user_request": request,
    }

    if extra_state:
        state.update(extra_state)

    result = graph.invoke(state)

    assert result["mode"] == mode
    assert result["evidence"]
    assert result["context"]
    assert result["result"] is not None

    print(f"  {mode}: PASSED")


def main() -> None:
    print("LANGGRAPH OFFLINE MULTI-MODE TEST")
    print("=" * 40)

    graph = build_study_graph(FakeAgent())

    test_mode(
        graph,
        "answer",
        "What is RAG?",
    )

    test_mode(
        graph,
        "explain",
        "Explain RAG.",
    )

    test_mode(
        graph,
        "summarize",
        "Summarize RAG.",
    )

    test_mode(
        graph,
        "quiz",
        "RAG",
        {
            "number_of_questions": 1,
        },
    )

    test_mode(
        graph,
        "flashcards",
        "RAG",
        {
            "number_of_cards": 1,
        },
    )

    test_mode(
        graph,
        "compare",
        "Compare RAG and fine-tuning.",
        {
            "topic_a": "RAG",
            "topic_b": "Fine-tuning",
        },
    )

    print("\nALL SIX MODES PASSED")
    print("OFFLINE LANGGRAPH MULTI-MODE TEST PASSED")


if __name__ == "__main__":
    main()