from src.embedding import GeminiEmbedder
from src.vectorstore import ChromaVectorStore


def main() -> None:
    question = "What is Retrieval-Augmented Generation?"

    embedder = GeminiEmbedder()
    store = ChromaVectorStore()

    query_embedding = embedder.embed_query(
        question
    )

    evidence = store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    print()
    print("=" * 70)
    print("STRUCTURED EVIDENCE TEST")
    print("=" * 70)

    print(f"Question: {question}")
    print(f"Evidence items: {len(evidence)}")

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        print()
        print("-" * 70)
        print(f"EVIDENCE {index}")
        print("-" * 70)

        print(f"Chunk ID: {item.chunk_id}")
        print(f"Source: {item.source}")
        print(f"Page: {item.page}")
        print(f"Distance: {item.distance:.4f}")
        print(f"Citation: {item.citation()}")

        print()
        print("Content:")
        print(item.content[:300])


if __name__ == "__main__":
    main()