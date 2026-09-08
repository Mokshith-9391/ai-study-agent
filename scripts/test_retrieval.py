from src.embedding import GeminiEmbedder
from src.vectorstore import ChromaVectorStore


def main() -> None:
    query = "What is Retrieval-Augmented Generation?"

    print()
    print("=" * 60)
    print("SEMANTIC RETRIEVAL TEST")
    print("=" * 60)

    print(f"Query: {query}")

    # Create query embedding
    embedder = GeminiEmbedder()
    query_embedding = embedder.embed_query(query)

    print(
        f"Query embedding dimensions: "
        f"{len(query_embedding)}"
    )

    # Search Chroma
    store = ChromaVectorStore()

    results = store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print()
    print(f"Results returned: {len(ids)}")

    for index, (
        chunk_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        print()
        print("-" * 60)
        print(f"RESULT {index}")
        print("-" * 60)

        print(f"Chunk ID: {chunk_id}")
        print(f"Source: {metadata.get('source')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Distance: {distance}")

        print()
        print("Content:")
        print(document[:800])


if __name__ == "__main__":
    main()