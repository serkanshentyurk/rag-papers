"""Generate a small synthetic evaluation set from the corpus, then measure
retrieval recall@k against it.

Idea: for a random sample of chunks, ask the LLM to write a question that the
chunk answers (in DIFFERENT words, so retrieval is tested on meaning, not
shared vocabulary). Each question's source chunk is known, so recall@k is
exact: run the question through retrieval, check whether the source chunk is
in the top-k.

This is a fast, honest bootstrap eval. Its known limitation: LLM-generated
questions skew toward the easily-answerable, so it tests retrieval and basic
grounding, not hard reasoning. A production eval would add human-written hard
cases.

Usage (from project root, after building embeddings):
    python -m scripts.build_eval
"""
import json
import random
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from src.embed import load_corpus
from src.store import build_index
from src.retrieve import retrieve

client = OpenAI()

_GEN_PROMPT = """You are helping build an evaluation set for a document search \
system. Read the passage below and write ONE specific question that this \
passage answers.

Rules:
- The question must be answerable from THIS passage alone.
- Use DIFFERENT wording from the passage where you can — do not copy its \
phrases. The goal is to test whether search finds this passage by meaning, \
not by matching words.
- Make it a real question a researcher might ask, not a vague one.
- Also give a short answer, drawn only from the passage.

Return ONLY valid JSON: {{"question": "...", "answer": "..."}}

Passage:
{chunk}"""


def generate_qa(chunk: str, model: str = "gpt-4o-mini") -> dict | None:
    """Ask the LLM for one (question, answer) pair grounded in the chunk."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _GEN_PROMPT.format(chunk=chunk)}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    # strip markdown fences if the model added them
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
        if "question" in obj and "answer" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


def build_eval_set(chunks: list[str], n: int = 10, seed: int = 0) -> list[dict]:
    """Sample n chunks and generate a Q-A pair for each, tagged with the
    source chunk index (so recall@k can be measured exactly)."""
    random.seed(seed)
    # sample chunks long enough to contain a real answerable fact
    candidates = [i for i, c in enumerate(chunks) if len(c) > 200]
    sample_idx = random.sample(candidates, min(n, len(candidates)))

    eval_set = []
    for idx in sample_idx:
        qa = generate_qa(chunks[idx])
        if qa:
            qa["source_index"] = idx
            eval_set.append(qa)
            print(f"  chunk {idx}: {qa['question']}")
    return eval_set


def measure_recall(eval_set: list[dict], index, chunks: list[str],
                   k: int = 5) -> float:
    """For each question, retrieve top-k and check whether the source chunk
    is among them. Returns recall@k (fraction of questions whose source chunk
    was retrieved)."""
    hits = 0
    for item in eval_set:
        retrieved = retrieve(item["question"], index, chunks, k=k)
        # match by text identity: is the source chunk among the retrieved?
        source_text = chunks[item["source_index"]]
        if source_text in retrieved:
            hits += 1
        else:
            print(f"  MISS: {item['question'][:70]}")
    return hits / len(eval_set) if eval_set else 0.0


def main() -> None:
    vectors, chunks = load_corpus()
    index = build_index(vectors)

    print("Generating evaluation set...")
    eval_set = build_eval_set(chunks, n=10)
    with open("data/eval_set.json", "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"\nGenerated {len(eval_set)} Q-A pairs (saved to data/eval_set.json)")

    print(f"\nMeasuring recall@5...")
    recall = measure_recall(eval_set, index, chunks, k=5)
    print(f"\nRecall@5: {recall:.2f}  "
          f"({int(recall * len(eval_set))}/{len(eval_set)} questions "
          f"retrieved their source chunk in the top 5)")


if __name__ == "__main__":
    main()