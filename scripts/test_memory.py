from pathlib import Path
import tempfile

from src.memory.database import StudyDatabase
from src.memory.repository import MemoryRepository


def main() -> None:
    print("=" * 70)
    print("MEMORY / LEARNING STATE TEST")
    print("=" * 70)

    database = None

    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "test.db"

        try:
            # ========================================================
            # DATABASE
            # ========================================================

            print()
            print("Creating temporary database...")

            database = StudyDatabase(database_path)
            memory = MemoryRepository(database)

            print(f"  Database: {database_path}")
            print("  Temporary database: PASSED")

            # ========================================================
            # SESSION
            # ========================================================

            print()
            print("Creating study session...")

            session = memory.start_session()

            assert session is not None
            assert session.id is not None

            session_id = session.id

            print(f"  Session ID: {session_id}")
            print("  Session creation: PASSED")

            # ========================================================
            # MESSAGES
            # ========================================================

            print()
            print("Adding messages...")

            user_content = (
                "What is Retrieval-Augmented Generation?"
            )

            assistant_content = (
                "Retrieval-Augmented Generation combines "
                "retrieval with generation."
            )

            memory.add_message(
                session_id=session_id,
                role="user",
                content=user_content,
            )

            memory.add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_content,
            )

            messages = memory.get_messages(session_id)

            assert len(messages) == 2
            assert messages[0].role == "user"
            assert messages[0].content == user_content
            assert messages[1].role == "assistant"
            assert messages[1].content == assistant_content

            print(f"  Messages stored: {len(messages)}")
            print("  Message persistence: PASSED")

            # ========================================================
            # QUIZ
            # ========================================================

            print()
            print("Recording quiz attempts...")

            memory.record_quiz_attempt(
                session_id=session_id,
                topic="RAG",
                question="What does RAG stand for?",
                selected_answer="Retrieval-Augmented Generation",
                correct_answer="Retrieval-Augmented Generation",
                is_correct=True,
            )

            memory.record_quiz_attempt(
                session_id=session_id,
                topic="RAG",
                question="Does RAG use retrieval?",
                selected_answer="No",
                correct_answer="Yes",
                is_correct=False,
            )

            quiz_progress = memory.get_topic_progress("RAG")

            assert quiz_progress is not None
            assert quiz_progress.quiz_correct == 1
            assert quiz_progress.quiz_total == 2

            print(
                f"  Quiz progress: "
                f"{quiz_progress.quiz_correct}/"
                f"{quiz_progress.quiz_total}"
            )

            print("  Quiz progress tracking: PASSED")

            # ========================================================
            # FLASHCARDS
            # ========================================================

            print()
            print("Recording flashcard reviews...")

            memory.record_flashcard_review(
                session_id=session_id,
                topic="RAG",
                front="What does RAG stand for?",
                remembered=True,
            )

            memory.record_flashcard_review(
                session_id=session_id,
                topic="RAG",
                front="What is the purpose of retrieval?",
                remembered=False,
            )

            flashcard_progress = (
                memory.get_topic_progress("RAG")
            )

            assert flashcard_progress is not None
            assert (
                flashcard_progress.flashcard_remembered
                == 1
            )
            assert (
                flashcard_progress.flashcard_total
                == 2
            )

            print(
                f"  Flashcard progress: "
                f"{flashcard_progress.flashcard_remembered}/"
                f"{flashcard_progress.flashcard_total}"
            )

            print(
                "  Flashcard progress tracking: PASSED"
            )

            # ========================================================
            # ALL TOPIC PROGRESS
            # ========================================================

            print()
            print("Checking all topic progress...")

            all_progress = memory.get_all_topic_progress()

            assert len(all_progress) >= 1

            rag_progress = next(
                (
                    item
                    for item in all_progress
                    if item.topic == "RAG"
                ),
                None,
            )

            assert rag_progress is not None

            assert rag_progress.quiz_correct == 1
            assert rag_progress.quiz_total == 2
            assert (
                rag_progress.flashcard_remembered
                == 1
            )
            assert (
                rag_progress.flashcard_total
                == 2
            )

            print(
                f"  Topics tracked: {len(all_progress)}"
            )

            print(
                "  Topic progress retrieval: PASSED"
            )

            # ========================================================
            # END SESSION
            # ========================================================

            print()
            print("Ending study session...")

            memory.end_session(session_id)

            print("  Session ending: PASSED")

            # ========================================================
            # CLOSE + REOPEN
            # ========================================================

            database.close()
            database = None

            print()
            print("Reopening database...")

            database = StudyDatabase(database_path)
            memory = MemoryRepository(database)

            # ========================================================
            # PERSISTENCE
            # ========================================================

            persisted_messages = memory.get_messages(
                session_id
            )

            assert len(persisted_messages) == 2
            assert (
                persisted_messages[0].content
                == user_content
            )
            assert (
                persisted_messages[1].content
                == assistant_content
            )

            persisted_progress = (
                memory.get_topic_progress("RAG")
            )

            assert persisted_progress is not None
            assert (
                persisted_progress.quiz_correct == 1
            )
            assert (
                persisted_progress.quiz_total == 2
            )
            assert (
                persisted_progress.flashcard_remembered
                == 1
            )
            assert (
                persisted_progress.flashcard_total
                == 2
            )

            print(
                "  Messages persisted after reopen: PASSED"
            )

            print(
                "  Learning progress persisted after reopen: PASSED"
            )

            # ========================================================
            # SUCCESS
            # ========================================================

            print()
            print("=" * 70)
            print("ALL MEMORY TESTS PASSED")
            print("=" * 70)

        finally:
            if database is not None:
                database.close()


if __name__ == "__main__":
    main()