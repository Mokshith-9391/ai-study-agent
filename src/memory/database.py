from pathlib import Path
import sqlite3
from threading import RLock

from src.config import settings


DEFAULT_DATABASE_PATH = settings.database_path


class StudyDatabase:
    """
    SQLite database for persistent study-agent state.

    Designed for Streamlit's execution model:
    - check_same_thread=False permits access from Streamlit threads.
    - RLock serializes access to the shared SQLite connection.
    - initialize() creates the current schema and migrates missing columns.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = RLock()

        self.connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self.initialize()

    @property
    def lock(self) -> RLock:
        return self._lock

    def initialize(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ended_at TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    topic TEXT NOT NULL,
                    question TEXT NOT NULL,
                    selected_answer TEXT NOT NULL,
                    correct_answer TEXT NOT NULL,
                    is_correct INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS flashcard_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    topic TEXT NOT NULL,
                    front TEXT NOT NULL,
                    remembered INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS topic_progress (
                    topic TEXT PRIMARY KEY,
                    quiz_correct INTEGER NOT NULL DEFAULT 0,
                    quiz_total INTEGER NOT NULL DEFAULT 0,
                    flashcard_remembered INTEGER NOT NULL DEFAULT 0,
                    flashcard_total INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # --------------------------------------------------------
            # Migrations for databases created by earlier versions.
            # --------------------------------------------------------

            self._ensure_column(
                "sessions",
                "created_at",
                "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            )

            self._ensure_column(
                "sessions",
                "ended_at",
                "TEXT",
            )

            self._ensure_column(
                "messages",
                "session_id",
                "INTEGER",
            )

            self._ensure_column(
                "messages",
                "role",
                "TEXT",
            )

            self._ensure_column(
                "messages",
                "content",
                "TEXT",
            )

            self._ensure_column(
                "messages",
                "created_at",
                "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            )

            self._ensure_column(
                "quiz_attempts",
                "session_id",
                "INTEGER",
            )

            self._ensure_column(
                "quiz_attempts",
                "topic",
                "TEXT",
            )

            self._ensure_column(
                "quiz_attempts",
                "question",
                "TEXT",
            )

            self._ensure_column(
                "quiz_attempts",
                "selected_answer",
                "TEXT",
            )

            self._ensure_column(
                "quiz_attempts",
                "correct_answer",
                "TEXT",
            )

            self._ensure_column(
                "quiz_attempts",
                "is_correct",
                "INTEGER NOT NULL DEFAULT 0",
            )

            self._ensure_column(
                "quiz_attempts",
                "created_at",
                "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            )

            self._ensure_column(
                "flashcard_reviews",
                "session_id",
                "INTEGER",
            )

            self._ensure_column(
                "flashcard_reviews",
                "topic",
                "TEXT",
            )

            self._ensure_column(
                "flashcard_reviews",
                "front",
                "TEXT",
            )

            self._ensure_column(
                "flashcard_reviews",
                "remembered",
                "INTEGER NOT NULL DEFAULT 0",
            )

            self._ensure_column(
                "flashcard_reviews",
                "created_at",
                "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            )

            self._ensure_column(
                "topic_progress",
                "quiz_correct",
                "INTEGER NOT NULL DEFAULT 0",
            )

            self._ensure_column(
                "topic_progress",
                "quiz_total",
                "INTEGER NOT NULL DEFAULT 0",
            )

            self._ensure_column(
                "topic_progress",
                "flashcard_remembered",
                "INTEGER NOT NULL DEFAULT 0",
            )

            self._ensure_column(
                "topic_progress",
                "flashcard_total",
                "INTEGER NOT NULL DEFAULT 0",
            )

            self._ensure_column(
                "topic_progress",
                "updated_at",
                "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            )

            self.connection.commit()

    def _ensure_column(
        self,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        cursor = self.connection.execute(
            f"PRAGMA table_info({table_name})"
        )

        existing_columns = {
            row["name"]
            for row in cursor.fetchall()
        }

        if column_name not in existing_columns:
            self.connection.execute(
                f"""
                ALTER TABLE {table_name}
                ADD COLUMN {column_name} {column_definition}
                """
            )

    def close(self) -> None:
        with self._lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()
