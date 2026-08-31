import json
import numpy as np
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from environment

_MODEL = "text-embedding-3-small"   # 1536 dims, cheap, unit-normed. -large is overkill here.
_BATCH = 100                        # send in batches to stay under the request limit


def embed_texts(texts: list[str], model: str = _MODEL) -> list[list[float]]:
    """
    Embed a list of strings, batched. API returns vectors in input order,
    which is how each vector stays paired with its source text.
    
    Args:
        texts (list[str]): A list of strings to embed.
        model (str): The embedding model to use. Defaults to "text-embedding-3-small".
    
    Returns:
        list[list[float]]: A list of embedding vectors corresponding to the input texts.
    """
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend(item.embedding for item in resp.data)
    return out


def build_corpus(chunks: list[str],
                 vec_path: str = "data/vectors.npy",
                 txt_path: str = "data/chunks.json") -> np.ndarray:
    """
    Embed all chunks once and persist vectors + texts (same order).
    Re-run only when the corpus changes; otherwise load_corpus.
    
    Args:
        chunks (list[str]): A list of text chunks to embed and save.
        vec_path (str): The file path to save the embedding vectors as a .npy file.
        txt_path (str): The file path to save the text chunks as a .json file.

    Returns:
        np.ndarray: An array of embedding vectors corresponding to the input text chunks.
    """
    vectors = np.array(embed_texts(chunks))
    np.save(vec_path, vectors)
    with open(txt_path, "w") as f:
        json.dump(chunks, f)
    return vectors


def load_corpus(vec_path: str = "data/vectors.npy",
                txt_path: str = "data/chunks.json") -> tuple[np.ndarray, list[str]]:
    """
    Load persisted vectors and their paired texts. vectors[i] <-> chunks[i].
    
    Args:
        vec_path (str): The file path to load the embedding vectors from a .npy file.
        txt_path (str): The file path to load the text chunks from a .json file.

    Returns:
        tuple[np.ndarray, list[str]]: A tuple containing the loaded embedding vectors and their corresponding text chunks.
    """
    vectors = np.load(vec_path)
    with open(txt_path) as f:
        chunks = json.load(f)
    return vectors, chunks