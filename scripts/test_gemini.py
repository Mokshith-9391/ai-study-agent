from dotenv import load_dotenv
from google import genai
import os


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. "
            "Check your .env file."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain RAG in exactly three simple sentences.",
    )

    print("\n=== Gemini Response ===\n")
    print(response.text)


if __name__ == "__main__":
    main()