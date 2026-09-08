from dataclasses import dataclass


@dataclass
class RetrievalCase:
    """
    A deterministic retrieval evaluation case.

    expected_sources contains source paths that should appear
    somewhere in the retrieved evidence.
    """

    question: str
    expected_sources: list[str]


@dataclass
class RetrievalResult:
    question: str
    retrieved_sources: list[str]
    expected_sources: list[str]
    hit: bool
    reciprocal_rank: float


def reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Return reciprocal rank of the first relevant source.

    Example:

        relevant source at position 1 → 1.0
        position 2 → 0.5
        position 3 → 0.333...
        no hit → 0.0
    """

    expected = set(expected_sources)

    for position, source in enumerate(
        retrieved_sources,
        start=1,
    ):
        if source in expected:
            return 1.0 / position

    return 0.0


def evaluate_retrieval_case(
    case: RetrievalCase,
    retrieved_sources: list[str],
) -> RetrievalResult:

    score = reciprocal_rank(
        retrieved_sources=retrieved_sources,
        expected_sources=case.expected_sources,
    )

    return RetrievalResult(
        question=case.question,
        retrieved_sources=retrieved_sources,
        expected_sources=case.expected_sources,
        hit=score > 0,
        reciprocal_rank=score,
    )


def hit_rate(
    results: list[RetrievalResult],
) -> float:

    if not results:
        return 0.0

    hits = sum(
        result.hit
        for result in results
    )

    return hits / len(results)


def mean_reciprocal_rank(
    results: list[RetrievalResult],
) -> float:

    if not results:
        return 0.0

    return sum(
        result.reciprocal_rank
        for result in results
    ) / len(results)