"""Interactive RAG over the paper corpus.

Loads the pre-built embeddings (run scripts/build_embeddings.py first),
builds the search index, then answers questions typed at the prompt.

Usage (from project root):
    python main.py
"""
from dotenv import load_dotenv

load_dotenv()   # load OPENAI_API_KEY before any OpenAI client is created

from src.embed import load_corpus
from src.store import build_index
from src.generate import rag


def main() -> None:
    try:
        vectors, chunks = load_corpus()
    except FileNotFoundError:
        raise SystemExit(
            "No embeddings found. Run:  python -m scripts.build_embeddings"
        )

    index = build_index(vectors)
    print(f"Loaded {len(chunks)} chunks. Ask a question about the papers "
          f"(empty line or Ctrl-C to quit).\n")

    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            print("Bye.")
            break
        answer = rag(query, index, chunks)
        print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()