from pathlib import Path

import chromadb

from src.evidence import Evidence
from src.config import settings


VECTORSTORE_PATH = settings.vectorstore_path
COLLECTION_NAME = "study_documents"


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path = VECTORSTORE_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(
        self,
        chunks,
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        ids = [
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ]

        documents = [
            chunk.page_content
            for chunk in chunks
        ]

        metadatas = [
            chunk.metadata
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[Evidence]:
        """
        Search the vector store and return
        structured evidence objects.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        available = self.count()
        if available == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, available),
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        evidence = []

        for (
            chunk_id,
            document,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            page = metadata.get("page_label")

            if page is None:
                raw_page = metadata.get("page")

                if raw_page is not None:
                    page = str(int(raw_page) + 1)

            evidence.append(
                Evidence(
                    chunk_id=chunk_id,
                    content=document,
                    source=metadata.get(
                        "source",
                        "unknown",
                    ),
                    page=page,
                    distance=distance,
                )
            )

        return evidence
