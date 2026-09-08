from src.evidence import Evidence
from src.rag_result import RAGResult


def main() -> None:
    evidence = [
        Evidence(
            chunk_id="abc123",
            content="RAG retrieves relevant chunks.",
            source="machine-learning/Rag.pdf",
            page="8",
            distance=0.45,
        )
    ]

    result = RAGResult(
        question="What is RAG?",
        answer="RAG retrieves relevant information [E1].",
        evidence=evidence,
    )

    print()
    print("=" * 70)
    print("RAG RESULT TEST")
    print("=" * 70)

    print()
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print(f"Evidence count: {len(result.evidence)}")

    print()
    print("First evidence:")
    print(result.evidence[0].citation())


if __name__ == "__main__":
    main()