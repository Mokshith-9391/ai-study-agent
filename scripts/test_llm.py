from src.llm import GeminiLLM


def main() -> None:
    llm = GeminiLLM()

    prompt = (
        "Explain Retrieval-Augmented Generation "
        "in exactly three simple sentences."
    )

    response = llm.generate(prompt)

    print()
    print("=" * 60)
    print("LLM TEST")
    print("=" * 60)

    print(f"Model: {llm.model}")

    print()
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()