import faiss
import numpy as np


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build an exact inner-product index over the chunk vectors.

    IndexFlatIP: 'Flat' = exact (checks every vector, right for ~500),
    'IP' = inner product = dot product = cosine similarity FOR UNIT-LENGTH
    vectors (OpenAI embeddings are unit-normed, so dot product == cosine).
    
    Args:
        vectors (np.ndarray): A 2D array of shape (num_chunks, embedding_dim)
            containing the vector embeddings for each chunk.
            
    Returns:
        faiss.Index: A FAISS index that can be used for efficient similarity search.
    """
    dim = vectors.shape[1]                
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype("float32"))   # FAISS requires float32
    return index