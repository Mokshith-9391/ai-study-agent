from src.embedding import GeminiEmbedder
from src.vectorstore import ChromaVectorStore
from src.context import build_context


def main() -> None:
    question = "What is Retrieval-Augmented Generation?"

    embedder = GeminiEmbedder()
    store = ChromaVectorStore()

    query_embedding = embedder.embed_query(question)

    evidence = store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    context = build_context(evidence)

    print()
    print("=" * 70)
    print("CONTEXT BUILDER TEST")
    print("=" * 70)

    print(f"Question: {question}")
    print(f"Evidence items: {len(evidence)}")
    print(f"Context characters: {len(context)}")

    print()
    print("-" * 70)
    print("GENERATED CONTEXT")
    print("-" * 70)

    print(context)


if __name__ == "__main__":
    main()