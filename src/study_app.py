from src.study_agent import StudyAgent


class StudyApp:
    """
    Thin application layer between the UI and StudyAgent.

    Keeps Streamlit-specific code out of the core agent.
    """

    def __init__(
        self,
        agent: StudyAgent | None = None,
    ) -> None:
        self.agent = agent or StudyAgent()

    def ask(self, question: str):
        return self.agent.answer(question)

    def explain(self, topic: str):
        return self.agent.explain(topic)

    def summarize(self, topic: str):
        return self.agent.summarize(topic)

    def quiz(
        self,
        topic: str,
        number_of_questions: int = 5,
    ):
        return self.agent.quiz(
            topic,
            number_of_questions,
        )

    def flashcards(
        self,
        topic: str,
        number_of_cards: int = 5,
    ):
        return self.agent.flashcards(
            topic,
            number_of_cards,
        )

    def compare(
        self,
        topic_a: str,
        topic_b: str,
    ):
        return self.agent.compare(
            topic_a,
            topic_b,
        )

    def start_session(self):
        return self.agent.start_session()

    def end_session(self):
        self.agent.end_session()

    def progress(self):
        return self.agent.get_all_progress()