from src.evaluation.metrics import (
    RetrievalCase,
    RetrievalResult,
    evaluate_retrieval_case,
    hit_rate,
    mean_reciprocal_rank,
)
from src.vectorstore import ChromaVectorStore
from src.embedding import GeminiEmbedder


class RetrievalEvaluator:
    """
    Evaluates the retrieval layer independently of the LLM.

    This is important because retrieval quality can be measured
    without generating an answer.
    """

    def __init__(
        self,
        embedder=None,
        vectorstore=None,
    ):
        self.embedder = embedder or GeminiEmbedder()
        self.vectorstore = vectorstore or ChromaVectorStore()

    def evaluate_case(
        self,
        case: RetrievalCase,
        top_k: int = 5,
    ) -> RetrievalResult:

        evidence = self._retrieve(
            case.question,
            top_k,
        )

        sources = [
            item.source
            for item in evidence
        ]

        return evaluate_retrieval_case(
            case=case,
            retrieved_sources=sources,
        )

    def evaluate(
        self,
        cases: list[RetrievalCase],
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        results = []

        for case in cases:
            result = self.evaluate_case(
                case=case,
                top_k=top_k,
            )

            results.append(result)

        return results

    def _retrieve(
        self,
        question: str,
        top_k: int,
    ):
        if not question.strip():
            raise ValueError(
                "Evaluation question must not be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if self.vectorstore.count() == 0:
            raise RuntimeError(
                "Vector store is empty."
            )

        embedding = self.embedder.embed_query(
            question
        )

        return self.vectorstore.search(
            query_embedding=embedding,
            top_k=top_k,
        )