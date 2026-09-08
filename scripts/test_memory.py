from pathlib import Path
import tempfile

from src.memory.database import StudyDatabase
from src.memory.repository import MemoryRepository


def main():
    print("=" * 70)
    print("MEMORY / LEARNING STATE TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "test.db"

        database = StudyDatabase(
            path=database_path,
        )

        memory = MemoryRepository(
            database=database,
        )

        # ---------------------------------------------------------
        # Session
        # ---------------------------------------------------------

        session = memory.start_session()

        assert session.id > 0

        print(f"\nSession created: {session.id}")

        # ---------------------------------------------------------
        # Conversation
        # ---------------------------------------------------------

        memory.add_message(
            session_id=session.id,
            role="user",
            content="What is RAG?",
        )

        memory.add_message(
            session_id=session.id,
            role="assistant",
            content="RAG combines retrieval with generation.",
        )

        messages = memory.get_messages(
            session_id=session.id,
        )

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

        print(f"Messages stored: {len(messages)}")

        # ---------------------------------------------------------
        # Quiz attempts
        # ---------------------------------------------------------

        memory.record_quiz_attempt(
            session_id=session.id,
            topic="RAG",
            question="What does RAG stand for?",
            selected_answer="Retrieval-Augmented Generation",
            correct_answer="Retrieval-Augmented Generation",
        )

        memory.record_quiz_attempt(
            session_id=session.id,
            topic="RAG",
            question="Does RAG retrieve external context?",
            selected_answer="No",
            correct_answer="Yes",
        )

        progress = memory.get_topic_progress("RAG")

        assert progress is not None
        assert progress.questions_answered == 2
        assert progress.questions_correct == 1
        assert progress.quiz_accuracy == 0.5

        print(
            f"Quiz progress: "
            f"{progress.questions_correct}/"
            f"{progress.questions_answered}"
        )

        # ---------------------------------------------------------
        # Flashcards
        # ---------------------------------------------------------

        memory.record_flashcard_review(
            session_id=session.id,
            topic="RAG",
            front="What is retrieval?",
            remembered=True,
        )

        memory.record_flashcard_review(
            session_id=session.id,
            topic="RAG",
            front="What is generation?",
            remembered=False,
        )

        progress = memory.get_topic_progress("RAG")

        assert progress.flashcards_reviewed == 2
        assert progress.flashcards_remembered == 1
        assert progress.flashcard_retention == 0.5

        print(
            f"Flashcard progress: "
            f"{progress.flashcards_remembered}/"
            f"{progress.flashcards_reviewed}"
        )

        # ---------------------------------------------------------
        # Persistence
        # ---------------------------------------------------------

        database.close()

        database = StudyDatabase(
            path=database_path,
        )

        memory = MemoryRepository(
            database=database,
        )

        progress = memory.get_topic_progress("RAG")

        assert progress is not None
        assert progress.questions_answered == 2
        assert progress.questions_correct == 1
        assert progress.flashcards_reviewed == 2
        assert progress.flashcards_remembered == 1

        messages = memory.get_messages(
            session_id=session.id,
        )

        assert len(messages) == 2

        print("\nPersistence check: PASSED")

        database.close()

    print("\n" + "=" * 70)
    print("ALL MEMORY TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()