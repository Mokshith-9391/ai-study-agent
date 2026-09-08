from datetime import datetime, timezone

from src.memory.database import StudyDatabase
from src.memory.models import (
    FlashcardReview,
    Message,
    QuizAttempt,
    StudySession,
    TopicProgress,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryRepository:
    """
    Repository for persistent learner memory.
    """

    def __init__(
        self,
        database: StudyDatabase | None = None,
    ) -> None:
        self.database = database or StudyDatabase()

    def start_session(self) -> StudySession:
        created_at = utc_now()

        cursor = self.database.connection.execute(
            """
            INSERT INTO sessions (created_at)
            VALUES (?)
            """,
            (created_at,),
        )

        self.database.connection.commit()

        return StudySession(
            id=cursor.lastrowid,
            created_at=created_at,
        )

    def end_session(
        self,
        session_id: int,
    ) -> None:
        self.database.connection.execute(
            """
            UPDATE sessions
            SET ended_at = ?
            WHERE id = ?
            """,
            (
                utc_now(),
                session_id,
            ),
        )

        self.database.connection.commit()

    def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
    ) -> Message:
        if role not in {
            "user",
            "assistant",
            "system",
        }:
            raise ValueError(
                "role must be user, assistant, or system."
            )

        created_at = utc_now()

        cursor = self.database.connection.execute(
            """
            INSERT INTO messages (
                session_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                created_at,
            ),
        )

        self.database.connection.commit()

        return Message(
            id=cursor.lastrowid,
            session_id=session_id,
            role=role,
            content=content,
            created_at=created_at,
        )

    def get_messages(
        self,
        session_id: int,
    ) -> list[Message]:
        rows = self.database.connection.execute(
            """
            SELECT
                id,
                session_id,
                role,
                content,
                created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

        return [
            Message(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def record_quiz_attempt(
        self,
        topic: str,
        question: str,
        selected_answer: str,
        correct_answer: str,
        session_id: int | None = None,
    ) -> QuizAttempt:
        is_correct = selected_answer == correct_answer
        created_at = utc_now()

        cursor = self.database.connection.execute(
            """
            INSERT INTO quiz_attempts (
                session_id,
                topic,
                question,
                selected_answer,
                correct_answer,
                is_correct,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                topic,
                question,
                selected_answer,
                correct_answer,
                int(is_correct),
                created_at,
            ),
        )

        self.database.connection.commit()

        self._update_topic_quiz_progress(
            topic=topic,
            is_correct=is_correct,
            studied_at=created_at,
        )

        return QuizAttempt(
            id=cursor.lastrowid,
            session_id=session_id,
            topic=topic,
            question=question,
            selected_answer=selected_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
            created_at=created_at,
        )

    def _update_topic_quiz_progress(
        self,
        topic: str,
        is_correct: bool,
        studied_at: str,
    ) -> None:
        self.database.connection.execute(
            """
            INSERT INTO topic_progress (
                topic,
                questions_answered,
                questions_correct,
                last_studied_at
            )
            VALUES (?, 1, ?, ?)
            ON CONFLICT(topic)
            DO UPDATE SET
                questions_answered =
                    questions_answered + 1,
                questions_correct =
                    questions_correct + excluded.questions_correct,
                last_studied_at =
                    excluded.last_studied_at
            """,
            (
                topic,
                int(is_correct),
                studied_at,
            ),
        )

        self.database.connection.commit()

    def record_flashcard_review(
        self,
        topic: str,
        front: str,
        remembered: bool,
        session_id: int | None = None,
    ) -> FlashcardReview:
        created_at = utc_now()

        cursor = self.database.connection.execute(
            """
            INSERT INTO flashcard_reviews (
                session_id,
                topic,
                front,
                remembered,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                topic,
                front,
                int(remembered),
                created_at,
            ),
        )

        self.database.connection.commit()

        self._update_topic_flashcard_progress(
            topic=topic,
            remembered=remembered,
            studied_at=created_at,
        )

        return FlashcardReview(
            id=cursor.lastrowid,
            session_id=session_id,
            topic=topic,
            front=front,
            remembered=remembered,
            created_at=created_at,
        )

    def _update_topic_flashcard_progress(
        self,
        topic: str,
        remembered: bool,
        studied_at: str,
    ) -> None:
        self.database.connection.execute(
            """
            INSERT INTO topic_progress (
                topic,
                flashcards_reviewed,
                flashcards_remembered,
                last_studied_at
            )
            VALUES (?, 1, ?, ?)
            ON CONFLICT(topic)
            DO UPDATE SET
                flashcards_reviewed =
                    flashcards_reviewed + 1,
                flashcards_remembered =
                    flashcards_remembered + excluded.flashcards_remembered,
                last_studied_at =
                    excluded.last_studied_at
            """,
            (
                topic,
                int(remembered),
                studied_at,
            ),
        )

        self.database.connection.commit()

    def get_topic_progress(
        self,
        topic: str,
    ) -> TopicProgress | None:
        row = self.database.connection.execute(
            """
            SELECT
                topic,
                questions_answered,
                questions_correct,
                flashcards_reviewed,
                flashcards_remembered,
                last_studied_at
            FROM topic_progress
            WHERE topic = ?
            """,
            (topic,),
        ).fetchone()

        if row is None:
            return None

        return TopicProgress(
            topic=row["topic"],
            questions_answered=row["questions_answered"],
            questions_correct=row["questions_correct"],
            flashcards_reviewed=row["flashcards_reviewed"],
            flashcards_remembered=row["flashcards_remembered"],
            last_studied_at=row["last_studied_at"],
        )

    def get_all_topic_progress(
        self,
    ) -> list[TopicProgress]:
        rows = self.database.connection.execute(
            """
            SELECT
                topic,
                questions_answered,
                questions_correct,
                flashcards_reviewed,
                flashcards_remembered,
                last_studied_at
            FROM topic_progress
            ORDER BY last_studied_at DESC
            """
        ).fetchall()

        return [
            TopicProgress(
                topic=row["topic"],
                questions_answered=row["questions_answered"],
                questions_correct=row["questions_correct"],
                flashcards_reviewed=row["flashcards_reviewed"],
                flashcards_remembered=row["flashcards_remembered"],
                last_studied_at=row["last_studied_at"],
            )
            for row in rows
        ]