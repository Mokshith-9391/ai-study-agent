from src.evaluation.metrics import (
    RetrievalCase,
    evaluate_retrieval_case,
    hit_rate,
    mean_reciprocal_rank,
    reciprocal_rank,
)


def main():
    print("=" * 70)
    print("RETRIEVAL EVALUATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Reciprocal rank
    # ---------------------------------------------------------

    assert reciprocal_rank(
        ["wrong.pdf", "target.pdf"],
        ["target.pdf"],
    ) == 0.5

    assert reciprocal_rank(
        ["target.pdf"],
        ["target.pdf"],
    ) == 1.0

    assert reciprocal_rank(
        ["wrong.pdf"],
        ["target.pdf"],
    ) == 0.0

    print("\nReciprocal rank: PASSED")

    # ---------------------------------------------------------
    # Evaluation cases
    # ---------------------------------------------------------

    case_1 = RetrievalCase(
        question="What is RAG?",
        expected_sources=[
            "machine-learning/Rag.pdf"
        ],
    )

    result_1 = evaluate_retrieval_case(
        case_1,
        [
            "machine-learning/Rag.pdf",
            "cloud/cloud_notes.txt",
        ],
    )

    assert result_1.hit
    assert result_1.reciprocal_rank == 1.0

    case_2 = RetrievalCase(
        question="What is cloud computing?",
        expected_sources=[
            "cloud/cloud_notes.txt"
        ],
    )

    result_2 = evaluate_retrieval_case(
        case_2,
        [
            "machine-learning/Rag.pdf",
            "cloud/cloud_notes.txt",
        ],
    )

    assert result_2.hit
    assert result_2.reciprocal_rank == 0.5

    results = [
        result_1,
        result_2,
    ]

    assert hit_rate(results) == 1.0
    assert mean_reciprocal_rank(results) == 0.75

    print("Evaluation cases: PASSED")
    print("Hit rate: 1.00")
    print("MRR: 0.75")

    # ---------------------------------------------------------
    # No results
    # ---------------------------------------------------------

    assert hit_rate([]) == 0.0
    assert mean_reciprocal_rank([]) == 0.0

    print("Empty evaluation set: PASSED")

    print("\n" + "=" * 70)
    print("ALL EVALUATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()