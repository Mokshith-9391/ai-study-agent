from src.embedding import GeminiEmbedder


def main() -> None:
    embedder = GeminiEmbedder()

    text = (
        "Retrieval-Augmented Generation retrieves "
        "relevant information from documents before "
        "generating an answer."
    )

    vector = embedder.embed_documents([text])[0]

    print()
    print("=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    print("Model: gemini-embedding-2")
    print(f"Dimensions: {len(vector)}")
    print(f"First 10 values: {vector[:10]}")


if __name__ == "__main__":
    main()