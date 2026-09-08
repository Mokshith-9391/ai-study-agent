import math

from src.embedding import GeminiEmbedder


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(x * x for x in b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError(
            "Cannot calculate similarity with zero vector."
        )

    return dot_product / (
        magnitude_a * magnitude_b
    )


def main() -> None:
    embedder = GeminiEmbedder()

    texts = [
        "RAG retrieves information from a knowledge base.",
        "Retrieval augmented generation searches documents before answering.",
        "Python loops repeat a block of code.",
    ]

    vectors = []

    for text in texts:
        vector = embedder.embed_query(text)
        vectors.append(vector)

    print()
    print("=" * 60)
    print("SEMANTIC SIMILARITY TEST")
    print("=" * 60)

    print(f"Vector count: {len(vectors)}")
    print(f"Vector dimensions: {len(vectors[0])}")

    similarity_ab = cosine_similarity(
        vectors[0],
        vectors[1],
    )

    similarity_ac = cosine_similarity(
        vectors[0],
        vectors[2],
    )

    print()
    print(
        f"RAG ↔ RAG-like concept: {similarity_ab:.4f}"
    )
    print(
        f"RAG ↔ Python concept:    {similarity_ac:.4f}"
    )

    print()
    print("Expected:")
    print("RAG ↔ RAG-like concept should be higher.")
    print("RAG ↔ Python concept should be lower.")


if __name__ == "__main__":
    main()