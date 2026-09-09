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
    Repository for persistent learner memory and learning progress.

    Responsibilities:
    - Study session persistence
    - Conversation/message persistence
    - Quiz attempt persistence
    - Flashcard review persistence
    - Aggregated topic progress
    """

    def __init__(
        self,
        database: StudyDatabase | None = None,
    ) -> None:
        self.database = database or StudyDatabase()

    def start_session(self) -> StudySession:
        created_at = utc_now()

        with self.database.lock:
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
        with self.database.lock:
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

        content = content.strip()

        if not content:
            raise ValueError(
                "Message content must not be empty."
            )

        created_at = utc_now()

        with self.database.lock:
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
        with self.database.lock:
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
        is_correct: bool | None = None,
    ) -> QuizAttempt:
        """
        Record one quiz attempt and update aggregated topic progress.

        `is_correct` is optional for backwards compatibility.

        If omitted, correctness is calculated from:
            selected_answer == correct_answer

        If supplied, it must agree with the calculated result.
        """

        topic = topic.strip()
        question = question.strip()
        selected_answer = selected_answer.strip()
        correct_answer = correct_answer.strip()

        if not topic:
            raise ValueError("topic must not be empty.")

        if not question:
            raise ValueError("question must not be empty.")

        if not selected_answer:
            raise ValueError(
                "selected_answer must not be empty."
            )

        if not correct_answer:
            raise ValueError(
                "correct_answer must not be empty."
            )

        calculated_is_correct = (
            selected_answer == correct_answer
        )

        if is_correct is None:
            is_correct = calculated_is_correct
        elif bool(is_correct) != calculated_is_correct:
            raise ValueError(
                "is_correct does not match selected_answer "
                "and correct_answer."
            )

        created_at = utc_now()

        with self.database.lock:
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

            self.database.connection.execute(
                """
                INSERT INTO topic_progress (
                    topic,
                    quiz_correct,
                    quiz_total,
                    updated_at
                )
                VALUES (?, ?, 1, ?)
                ON CONFLICT(topic)
                DO UPDATE SET
                    quiz_correct =
                        topic_progress.quiz_correct
                        + excluded.quiz_correct,
                    quiz_total =
                        topic_progress.quiz_total
                        + 1,
                    updated_at =
                        excluded.updated_at
                """,
                (
                    topic,
                    int(is_correct),
                    created_at,
                ),
            )

            self.database.connection.commit()

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

    def record_flashcard_review(
        self,
        topic: str,
        front: str,
        remembered: bool,
        session_id: int | None = None,
    ) -> FlashcardReview:
        topic = topic.strip()
        front = front.strip()

        if not topic:
            raise ValueError("topic must not be empty.")

        if not front:
            raise ValueError(
                "front must not be empty."
            )

        created_at = utc_now()

        with self.database.lock:
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

            self.database.connection.execute(
                """
                INSERT INTO topic_progress (
                    topic,
                    flashcard_remembered,
                    flashcard_total,
                    updated_at
                )
                VALUES (?, ?, 1, ?)
                ON CONFLICT(topic)
                DO UPDATE SET
                    flashcard_remembered =
                        topic_progress.flashcard_remembered
                        + excluded.flashcard_remembered,
                    flashcard_total =
                        topic_progress.flashcard_total
                        + 1,
                    updated_at =
                        excluded.updated_at
                """,
                (
                    topic,
                    int(remembered),
                    created_at,
                ),
            )

            self.database.connection.commit()

        return FlashcardReview(
            id=cursor.lastrowid,
            session_id=session_id,
            topic=topic,
            front=front,
            remembered=remembered,
            created_at=created_at,
        )

    def get_topic_progress(
        self,
        topic: str,
    ) -> TopicProgress | None:
        topic = topic.strip()

        with self.database.lock:
            row = self.database.connection.execute(
                """
                SELECT
                    topic,
                    quiz_correct,
                    quiz_total,
                    flashcard_remembered,
                    flashcard_total,
                    updated_at
                FROM topic_progress
                WHERE topic = ?
                """,
                (topic,),
            ).fetchone()

        if row is None:
            return None

        return TopicProgress(
            topic=row["topic"],
            quiz_correct=row["quiz_correct"],
            quiz_total=row["quiz_total"],
            flashcard_remembered=row[
                "flashcard_remembered"
            ],
            flashcard_total=row["flashcard_total"],
            updated_at=row["updated_at"],
        )

    def get_all_topic_progress(
        self,
    ) -> list[TopicProgress]:
        with self.database.lock:
            rows = self.database.connection.execute(
                """
                SELECT
                    topic,
                    quiz_correct,
                    quiz_total,
                    flashcard_remembered,
                    flashcard_total,
                    updated_at
                FROM topic_progress
                ORDER BY updated_at DESC
                """
            ).fetchall()

        return [
            TopicProgress(
                topic=row["topic"],
                quiz_correct=row["quiz_correct"],
                quiz_total=row["quiz_total"],
                flashcard_remembered=row[
                    "flashcard_remembered"
                ],
                flashcard_total=row["flashcard_total"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]