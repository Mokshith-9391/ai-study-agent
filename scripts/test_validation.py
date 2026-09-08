from src.validation import (
    ValidationError,
    require_valid,
    validate_comparison,
    validate_flashcard_item,
    validate_quiz_item,
)


def main():
    print("=" * 70)
    print("CONTENT VALIDATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Valid quiz
    # ---------------------------------------------------------

    valid_quiz = {
        "question": "What does RAG stand for?",
        "options": [
            "Retrieval-Augmented Generation",
            "Random Answer Generation",
            "Rapid AI Generation",
            "Recursive Answer Graph",
        ],
        "answer": "Retrieval-Augmented Generation",
        "explanation": "RAG combines retrieval with generation.",
        "evidence": ["E1"],
    }

    result = validate_quiz_item(valid_quiz)

    assert result.valid

    print("\nValid quiz: PASSED")

    # ---------------------------------------------------------
    # Invalid quiz
    # ---------------------------------------------------------

    invalid_quiz = {
        "question": "What does RAG stand for?",
        "options": [
            "A",
            "B",
            "C",
        ],
        "answer": "D",
        "explanation": "Example",
        "evidence": ["E1"],
    }

    result = validate_quiz_item(invalid_quiz)

    assert not result.valid
    assert len(result.errors) >= 2

    print("Invalid quiz rejected: PASSED")

    # ---------------------------------------------------------
    # Valid flashcard
    # ---------------------------------------------------------

    valid_flashcard = {
        "front": "What is RAG?",
        "back": "Retrieval-Augmented Generation.",
        "evidence": ["E1"],
    }

    result = validate_flashcard_item(
        valid_flashcard
    )

    assert result.valid

    print("Valid flashcard: PASSED")

    # ---------------------------------------------------------
    # Invalid flashcard
    # ---------------------------------------------------------

    invalid_flashcard = {
        "front": "What is RAG?",
        "evidence": ["E1"],
    }

    result = validate_flashcard_item(
        invalid_flashcard
    )

    assert not result.valid

    print("Invalid flashcard rejected: PASSED")

    # ---------------------------------------------------------
    # Valid comparison
    # ---------------------------------------------------------

    valid_comparison = {
        "topic_a": "RAG",
        "topic_b": "Fine-tuning",
        "comparison": "They differ in how knowledge is incorporated.",
        "evidence": ["E1", "E2"],
    }

    result = validate_comparison(
        valid_comparison
    )

    assert result.valid

    print("Valid comparison: PASSED")

    # ---------------------------------------------------------
    # Invalid comparison
    # ---------------------------------------------------------

    invalid_comparison = {
        "topic_a": "RAG",
        "comparison": 123,
        "evidence": "E1",
    }

    result = validate_comparison(
        invalid_comparison
    )

    assert not result.valid

    print("Invalid comparison rejected: PASSED")

    # ---------------------------------------------------------
    # require_valid
    # ---------------------------------------------------------

    try:
        require_valid(
            validate_quiz_item(invalid_quiz)
        )
    except ValidationError:
        print("ValidationError: PASSED")
    else:
        raise AssertionError(
            "ValidationError was not raised."
        )

    print("\n" + "=" * 70)
    print("ALL VALIDATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()