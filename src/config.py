"""Central configuration for the local study agent."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    """Paths and provider settings that are safe to expose operationally."""

    study_folder: Path = _path_from_env("STUDY_FOLDER", "data/study")
    vectorstore_path: Path = _path_from_env("VECTORSTORE_PATH", "vectorstore")
    database_path: Path = _path_from_env("DATABASE_PATH", "data/study_agent.db")
    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"
    )
    gemini_generation_model: str = os.getenv(
        "GEMINI_GENERATION_MODEL", "gemini-3.8-flash"
    )


settings = Settings()
