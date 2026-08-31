from openai import OpenAI
from src.retrieve import retrieve
import faiss

client = OpenAI()   # reads OPENAI_API_KEY from environment


def build_prompt(query: str, chunks: list[str]) -> str:
    """Assemble the grounded prompt. Chunks are separated so the model sees
    them as distinct passages; the instruction enforces answer-from-context-
    only and explicit refusal when the answer is absent (the anti-hallucination
    guardrail).
    Args:
        query (str): The user's question.
        chunks (list[str]): The retrieved text chunks relevant to the query.
    Returns:
        str: The full prompt to send to the model.
    """
    context = "\n\n---\n\n".join(chunks)
    return f"""Answer the question using only the context below. If the answer \
is not in the context, say you cannot find it in the provided papers — do not \
use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call the chat endpoint. temperature=0 for deterministic, faithful-to-
    context answers rather than creative embellishment (creativity = more
    hallucination risk in RAG).
    Args:
        prompt (str): The full prompt to send to the model.
        model (str): The OpenAI model to use for generation.
    Returns:
        str: The model's answer to the query, grounded in the provided context.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content



def rag(query: str, index:faiss.Index, chunks: list[str], k: int = 5,
        model: str = "gpt-4o-mini") -> str:
    """The whole pipeline in three steps: Retrieve relevant chunks, Augment a
    prompt with them, Generate a grounded answer.
    Args:
        query (str): The user's question.
        index: The FAISS index built over the corpus vectors.
        chunks (list[str]): The text chunks corresponding to the vectors in the index.
        k (int): The number of nearest neighbors to retrieve for context.
        model (str): The OpenAI model to use for generation.
    Returns:
        str: The model's answer to the query, grounded in the retrieved context.
    """
    retrieved = retrieve(query, index, chunks, k=k)   # R
    prompt = build_prompt(query, retrieved)           # A
    return generate_answer(prompt, model=model)       # G