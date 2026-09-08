from src.citations import render_citations
from src.context import build_context
from src.embedding import GeminiEmbedder
from src.llm import GeminiLLM
from src.rag_result import RAGResult
from src.vectorstore import ChromaVectorStore
from src.evidence import Evidence


DEFAULT_TOP_K = 5


class RAGPipeline:
    def __init__(
        self,
        embedder=None,
        vectorstore=None,
        llm=None,
    ) -> None:
        self.embedder = embedder or GeminiEmbedder()
        self.vectorstore = vectorstore or ChromaVectorStore()
        self.llm = llm or GeminiLLM()

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[Evidence]:
        """
        Retrieve relevant evidence without calling the LLM.
        """

        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.vectorstore.count() == 0:
            raise RuntimeError(
                "The vector store is empty. "
                "Index study documents before asking questions."
            )

        query_embedding = self.embedder.embed_query(question)

        evidence = self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not evidence:
            raise RuntimeError("No relevant evidence was retrieved.")

        return evidence

    def build_context(
        self,
        evidence: list[Evidence],
    ) -> str:
        """
        Convert retrieved evidence into LLM-ready context.
        """
        return build_context(evidence)

    def generate_grounded_answer(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> RAGResult:
        """
        Generate a normal grounded answer from already-retrieved evidence.
        """

        context = self.build_context(evidence)

        prompt = f"""
You are a study assistant.

Answer the user's question using ONLY the supplied evidence.

USER QUESTION:
{question}

SUPPLIED EVIDENCE:
{context}

GROUNDING RULES:
1. Use only information supported by the evidence.
2. Do not invent facts.
3. Do not invent source names, page numbers, or chunk IDs.
4. When making a factual claim based on evidence, cite it using the evidence
   labels exactly as provided, such as [E1] or [E1, E3].
5. If the evidence does not contain enough information, say so clearly.
6. Prefer a clear educational explanation over unnecessary verbosity.

Return only the answer.
"""

        raw_answer = self.llm.generate(prompt)

        final_answer = render_citations(
            answer=raw_answer,
            evidence=evidence,
        )

        return RAGResult(
            question=question,
            answer=final_answer,
            evidence=evidence,
        )

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> RAGResult:
        """
        Full RAG operation:
        retrieve → context → generate → citations.
        """

        evidence = self.retrieve(
            question=question,
            top_k=top_k,
        )

        return self.generate_grounded_answer(
            question=question,
            evidence=evidence,
        )