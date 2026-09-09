from pathlib import Path
import tempfile

from src.memory.repository import MemoryRepository
from src.study_app import StudyApp


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "Offline test response [E1]"


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 768


class FakeVectorStore:
    def count(self) -> int:
        return 1

    def search(self, query_embedding, top_k: int):
        return []


def main() -> None:
    print("=" * 70)
    print("STUDY APP MEMORY WIRING TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "test.db"

        print()
        print("Creating isolated database...")

        repository = MemoryRepository()

        # Replace the repository's database with an isolated one.
        from src.memory.database import StudyDatabase

        database = StudyDatabase(database_path)
        repository = MemoryRepository(database)

        print("  Temporary database: PASSED")

        print()
        print("Creating StudyApp with injected repository...")

        app = StudyApp(
            memory_repository=repository,
        )

        print("  StudyApp creation: PASSED")

        print()
        print("Checking repository identity...")

        assert app.memory_repository is repository
        print("  StudyApp repository identity: PASSED")

        print()
        print("Checking StudyAgent repository identity...")

        assert app.agent.memory is repository
        print("  StudyAgent uses StudyApp repository: PASSED")

        print()
        print("Checking LangGraph repository wiring...")

        # The graph nodes are closures created with the same
        # repository. We verify this by running a session through
        # the StudyApp and checking persistence in the injected DB.

        session = app.start_session()

        assert session.id is not None
        assert app.session_id == session.id

        print(f"  Session ID: {session.id}")
        print("  Shared session ownership: PASSED")

        print()
        print("Checking session persistence in injected database...")

        row = database.connection.execute(
            """
            SELECT id
            FROM sessions
            WHERE id = ?
            """,
            (session.id,),
        ).fetchone()

        assert row is not None
        print("  Session stored in injected database: PASSED")

        print()
        print("Checking StudyAgent current session...")

        assert app.agent.current_session is not None
        assert app.agent.current_session.id == session.id

        print("  StudyAgent session identity: PASSED")

        print()
        print("Ending session...")

        app.end_session()

        assert app.session_id is None
        assert app.agent.current_session is None

        print("  StudyApp session cleared: PASSED")
        print("  StudyAgent session cleared: PASSED")

        database.close()

    print()
    print("=" * 70)
    print("ALL STUDY APP MEMORY WIRING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()