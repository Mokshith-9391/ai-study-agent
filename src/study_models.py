from dataclasses import dataclass

from src.evidence import Evidence


@dataclass
class QuizQuestion:
    question: str
    options: list[str]
    answer: str
    explanation: str
    evidence: list[Evidence]


@dataclass
class Flashcard:
    front: str
    back: str
    evidence: list[Evidence]


@dataclass
class Comparison:
    topic_a: str
    topic_b: str
    comparison: str
    evidence: list[Evidence]