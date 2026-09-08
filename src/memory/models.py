from dataclasses import dataclass


@dataclass
class StudySession:
    id: int
    created_at: str
    ended_at: str | None = None


@dataclass
class Message:
    id: int
    session_id: int
    role: str
    content: str
    created_at: str


@dataclass
class QuizAttempt:
    id: int
    session_id: int | None
    topic: str
    question: str
    selected_answer: str
    correct_answer: str
    is_correct: bool
    created_at: str


@dataclass
class FlashcardReview:
    id: int
    session_id: int | None
    topic: str
    front: str
    remembered: bool
    created_at: str


@dataclass
class TopicProgress:
    topic: str
    questions_answered: int
    questions_correct: int
    flashcards_reviewed: int
    flashcards_remembered: int
    last_studied_at: str | None

    @property
    def quiz_accuracy(self) -> float:
        if self.questions_answered == 0:
            return 0.0

        return (
            self.questions_correct
            / self.questions_answered
        )

    @property
    def flashcard_retention(self) -> float:
        if self.flashcards_reviewed == 0:
            return 0.0

        return (
            self.flashcards_remembered
            / self.flashcards_reviewed
        )