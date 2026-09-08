from src.embedding import GeminiEmbedder
from src.vectorstore import ChromaVectorStore


TEST_QUERIES = [
    {
        "name": "Why RAG is needed",
        "query": "Why do we need RAG if an LLM already knows information?",
        "expected_source": "machine-learning/Rag.pdf",
    },
    {
        "name": "Hallucinations",
        "query": "How can RAG help reduce incorrect or made-up answers?",
        "expected_source": "machine-learning/Rag.pdf",
    },
    {
        "name": "Wrong retrieval",
        "query": "What happens when the retrieval system brings the wrong information?",
        "expected_source": "machine-learning/Rag.pdf",
    },
    {
        "name": "Embedding purpose",
        "query": "Why do we convert documents into numerical vectors?",
        "expected_source": "machine-learning/Rag.pdf",
    },
    {
        "name": "Cloud question",
        "query": "What are the main benefits of using cloud computing?",
        "expected_source": "cloud/cloud_notes.txt",
    },
]


def main() -> None:
    embedder = GeminiEmbedder()
    store = ChromaVectorStore()

    print()
    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    for test in TEST_QUERIES:
        query = test["query"]
        expected_source = test["expected_source"]

        query_embedding = embedder.embed_query(query)

        results = store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print()
        print("=" * 70)
        print(f"TEST: {test['name']}")
        print("=" * 70)
        print(f"Query: {query}")
        print(f"Expected source: {expected_source}")

        print()

        for rank, (
            chunk_id,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                metadatas,
                distances,
            ),
            start=1,
        ):
            source = metadata.get("source")
            page = metadata.get("page")

            marker = (
                "✓"
                if source == expected_source
                else " "
            )

            print(
                f"{marker} Rank {rank} | "
                f"Distance: {distance:.4f} | "
                f"Source: {source} | "
                f"Page: {page} | "
                f"ID: {chunk_id}"
            )


if __name__ == "__main__":
    main()