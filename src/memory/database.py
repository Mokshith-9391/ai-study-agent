import sqlite3
from pathlib import Path


DATABASE_PATH = Path("data") / "study_agent.db"


class StudyDatabase:
    """
    SQLite database used for learner state and study history.

    This database stores structured learning information.
    It does NOT store document embeddings or source chunks.
    """

    def __init__(
        self,
        path: str | Path = DATABASE_PATH,
    ) -> None:
        self.path = Path(path)

        if self.path.parent:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.connection = sqlite3.connect(
            self.path,
        )

        self.connection.row_factory = sqlite3.Row

        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ended_at TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                selected_answer TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS flashcard_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                topic TEXT NOT NULL,
                front TEXT NOT NULL,
                remembered INTEGER NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS topic_progress (
                topic TEXT PRIMARY KEY,
                questions_answered INTEGER NOT NULL DEFAULT 0,
                questions_correct INTEGER NOT NULL DEFAULT 0,
                flashcards_reviewed INTEGER NOT NULL DEFAULT 0,
                flashcards_remembered INTEGER NOT NULL DEFAULT 0,
                last_studied_at TEXT
            );
            """
        )

        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()