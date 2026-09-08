from src.embedding import GeminiEmbedder
from src.vectorstore import ChromaVectorStore
from src.context import build_context
from src.llm import GeminiLLM
from src.citations import render_citations


def main() -> None:
    question = "What is Retrieval-Augmented Generation?"

    embedder = GeminiEmbedder()
    store = ChromaVectorStore()
    llm = GeminiLLM()

    # 1. Embed the user's question.
    query_embedding = embedder.embed_query(question)

    # 2. Retrieve structured evidence.
    evidence = store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    # 3. Build the context for the LLM.
    context = build_context(evidence)

    # 4. Ask the LLM to answer using evidence labels.
    prompt = f"""
You are a study assistant.

Answer the user's question using ONLY the provided study context.

Rules:
- Do not invent facts.
- Do not use outside knowledge.
- If the context is insufficient, clearly say so.
- When making a factual claim supported by the context, cite it
  using the evidence label format [E1], [E2], [E3], etc.
- Use only evidence labels that actually exist in the provided context.
- Do not write filenames, page numbers, or chunk IDs yourself.
- Do not create or modify citation metadata.

User question:
{question}

Study context:
{context}
""".strip()

    # 5. Generate the answer.
    raw_answer = llm.generate(prompt)

    # 6. Replace [E1], [E2], etc. with
    # application-controlled citations.
    final_answer = render_citations(
        answer=raw_answer,
        evidence=evidence,
    )

    print()
    print("=" * 70)
    print("END-TO-END RAG TEST")
    print("=" * 70)

    print()
    print("QUESTION")
    print("-" * 70)
    print(question)

    print()
    print("RAW LLM ANSWER")
    print("-" * 70)
    print(raw_answer)

    print()
    print("FINAL ANSWER WITH APPLICATION-CONTROLLED CITATIONS")
    print("-" * 70)
    print(final_answer)

    print()
    print("RETRIEVED EVIDENCE")
    print("-" * 70)

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        print(
            f"[E{index}] "
            f"{item.citation()} "
            f"(distance={item.distance:.4f})"
        )


if __name__ == "__main__":
    main()