import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a string using the cl100k_base encoding."""
    return len(_enc.encode(text))


def _split_paragraph_by_tokens(paragraph: str, max_tokens: int) -> list[str]:
    """Blunt fallback for a single paragraph that alone exceeds the budget.
    Slices by token count with no sentence-awareness — deliberately crude,
    because giant paragraphs are rare and perfect cut-points aren't worth it."""
    token_ids = _enc.encode(paragraph)
    pieces = []
    for i in range(0, len(token_ids), max_tokens):
        piece_ids = token_ids[i:i + max_tokens]
        pieces.append(_enc.decode(piece_ids))
    return pieces


def chunk_text(text: str, max_tokens: int = 400, overlap_paragraphs: int = 1) -> list[str]:
    """Split text into chunks by packing whole paragraphs up to a token budget.

    - Normal case: accumulate paragraphs until adding the next would exceed
      max_tokens, then close the chunk.
    - Oversized-paragraph case: a single paragraph larger than max_tokens is
      token-split on its own (bounded option b).
    - Overlap: each new chunk is seeded with the last `overlap_paragraphs`
      paragraphs of the previous one, so a fact split across a boundary
      survives intact in at least one chunk.
      
    Args:
        text (str): The input text to be chunked.
        max_tokens (int): The maximum number of tokens allowed in each chunk.
        overlap_paragraphs (int): The number of paragraphs to overlap between chunks.
        
    Returns:
        list[str]: A list of text chunks, each containing whole paragraphs and
        respecting the specified token limit and overlap.
    """
    # paragraphs = non-empty blocks separated by blank lines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # our cleaning joined kept lines with single "\n", so fall back to
    # single-newline splitting if blank-line splitting yields one big blob:
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []          # paragraphs in the chunk being built
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # --- oversized single paragraph: flush current, then split it alone ---
        if para_tokens > max_tokens:
            if current:                       # close whatever we were building
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_split_paragraph_by_tokens(para, max_tokens))
            continue

        # --- normal case: would adding this paragraph overflow the budget? ---
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            # seed the next chunk with overlap from the end of this one
            overlap = current[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            current = list(overlap)
            current_tokens = sum(count_tokens(p) for p in current)

        current.append(para)
        current_tokens += para_tokens

    if current:                                # don't lose the final chunk
        chunks.append("\n\n".join(current))

    return chunks