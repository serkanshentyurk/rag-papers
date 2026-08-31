import numpy as np
import faiss
from src.embed import embed_texts



def retrieve(query: str, index: faiss.Index, chunks: list[str], k: int = 5) -> list[str]:
    """Embed the query, find the k nearest chunk vectors, return their text.

    index.search returns positions of the nearest vectors; we map those
    positions back to text via the parallel `chunks` list. This is why
    vectors[i] and chunks[i] must stay in the same order.
    
    Args:
        query (str): The input query string to retrieve relevant chunks for.
        index (faiss.Index): A FAISS index built over the chunk vectors.
        chunks (list[str]): A list of text chunks corresponding to the vectors in the index.
        k (int): The number of nearest neighbors to retrieve. Defaults to 5.

    Returns:
        list[str]: A list of the k nearest text chunks to the query.
    """
    q_vec = np.array(embed_texts([query])).astype("float32")
    scores, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0]]