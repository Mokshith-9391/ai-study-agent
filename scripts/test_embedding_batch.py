from src.embedding import (
    GeminiEmbedder,
    EMBEDDING_DIMENSION,
)


def main() -> None:
    embedder = GeminiEmbedder()

    texts = [
        "RAG retrieves information from a knowledge base.",
        "Retrieval augmented generation searches documents before answering.",
        "Python loops repeat a block of code.",
    ]

    vectors = embedder.embed_documents(texts)

    print()
    print("=" * 60)
    print("BATCH EMBEDDING TEST")
    print("=" * 60)

    print(f"Input texts: {len(texts)}")
    print(f"Output vectors: {len(vectors)}")

    for index, vector in enumerate(vectors, start=1):
        print(
            f"Vector {index}: "
            f"{len(vector)} dimensions"
        )

    if len(vectors) != len(texts):
        raise RuntimeError(
            "Input/output count mismatch."
        )

    if any(
        len(vector) != EMBEDDING_DIMENSION
        for vector in vectors
    ):
        raise RuntimeError(
            "Embedding dimension mismatch."
        )

    print()
    print("Batch embedding test PASSED.")


if __name__ == "__main__":
    main()