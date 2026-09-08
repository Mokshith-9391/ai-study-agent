from src.study_agent import StudyAgent


def main() -> None:
    agent = StudyAgent()

    print()
    print("=" * 70)
    print("STUDY AGENT INTEGRATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # ANSWER
    # ---------------------------------------------------------

    print()
    print("1. ANSWER")
    print("-" * 70)

    answer = agent.answer(
        "What is Retrieval-Augmented Generation?"
    )

    print(answer.answer)

    # ---------------------------------------------------------
    # EXPLAIN
    # ---------------------------------------------------------

    print()
    print("2. EXPLAIN")
    print("-" * 70)

    explanation = agent.explain(
        "embeddings"
    )

    print(explanation.answer)

    # ---------------------------------------------------------
    # SUMMARIZE
    # ---------------------------------------------------------

    print()
    print("3. SUMMARIZE")
    print("-" * 70)

    summary = agent.summarize(
        "Retrieval-Augmented Generation"
    )

    print(summary.answer)

    # ---------------------------------------------------------
    # QUIZ
    # ---------------------------------------------------------

    print()
    print("4. QUIZ")
    print("-" * 70)

    quiz = agent.quiz(
        topic="Retrieval-Augmented Generation",
        number_of_questions=3,
    )

    for index, question in enumerate(
        quiz,
        start=1,
    ):
        print()
        print(f"Question {index}:")
        print(question.question)

        for option in question.options:
            print(f"  - {option}")

        print(f"Answer: {question.answer}")
        print(f"Explanation: {question.explanation}")

        print("Sources:")
        for item in question.evidence:
            print(f"  - {item.citation()}")

    # ---------------------------------------------------------
    # FLASHCARDS
    # ---------------------------------------------------------

    print()
    print("5. FLASHCARDS")
    print("-" * 70)

    cards = agent.flashcards(
        topic="embeddings",
        number_of_cards=3,
    )

    for index, card in enumerate(
        cards,
        start=1,
    ):
        print()
        print(f"Card {index}")
        print(f"Front: {card.front}")
        print(f"Back: {card.back}")

        print("Sources:")
        for item in card.evidence:
            print(f"  - {item.citation()}")

    # ---------------------------------------------------------
    # COMPARE
    # ---------------------------------------------------------

    print()
    print("6. COMPARE")
    print("-" * 70)

    comparison = agent.compare(
        topic_a="RAG",
        topic_b="fine-tuning",
    )

    print(
        f"{comparison.topic_a} vs "
        f"{comparison.topic_b}"
    )

    print()
    print(comparison.comparison)

    print()
    print("Sources:")

    for item in comparison.evidence:
        print(f"  - {item.citation()}")

    print()
    print("=" * 70)
    print("STUDY AGENT INTEGRATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()