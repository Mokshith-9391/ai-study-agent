from src.rag import RAGPipeline


def main() -> None:
    question = "What is Retrieval-Augmented Generation?"

    rag = RAGPipeline()

    result = rag.ask(
        question=question,
        top_k=5,
    )

    print()
    print("=" * 70)
    print("PRODUCTION RAG CORE TEST")
    print("=" * 70)

    print()
    print("QUESTION")
    print("-" * 70)
    print(result.question)

    print()
    print("ANSWER")
    print("-" * 70)
    print(result.answer)

    print()
    print("SOURCES")
    print("-" * 70)

    for source in result.sources:
        print(source)

    print()
    print("EVIDENCE")
    print("-" * 70)

    for index, item in enumerate(
        result.evidence,
        start=1,
    ):
        print(
            f"[E{index}] "
            f"{item.citation()} "
            f"(distance={item.distance:.4f})"
        )


if __name__ == "__main__":
    main()