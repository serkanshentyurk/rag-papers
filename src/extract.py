""" Extracts and cleans text from PDF files using PyMuPDF. """

import pymupdf 

def strip_references(text: str) -> str:
    """
    Removes reference sections from the extracted text.

    Args:
        text (str): The extracted text.

    Returns:
        str: The text with reference sections removed.
    """

    # look for common reference-section markers; cut from the last plausible one
    markers = ["\nReferences\n", "\nREFERENCES\n", "\nBibliography\n"]
    cut = len(text)
    for m in markers:
        idx = text.rfind(m)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]

def drop_junk_lines(text: str, min_len: int = 30, min_alpha_ratio: float = 0.5) -> str:
    """
    Removes lines that are too short or contain too few alphabetic characters.
    
    Args:
        text (str): The extracted text.
        min_len (int): Minimum length of a line to keep.
        min_alpha_ratio (float): Minimum ratio of alphabetic characters to total characters in a line
        
    Returns:
        str: The text with junk lines removed.
    """
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < min_len:
            continue                          # too short — headers, fragments, page numbers
        alpha = sum(c.isalpha() or c.isspace() for c in stripped)
        if alpha / len(stripped) < min_alpha_ratio:
            continue                          # mostly numbers/symbols — figure axis soup
        kept.append(stripped)
    return "\n".join(kept)

def extract_text(pdf_path: str) -> str:
    """
    Extracts and cleans text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted and cleaned text from the PDF file.
    """
    doc = pymupdf.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    text = text.replace("-\n", "")      # join hyphenated line-wraps: "selec-\ntivity" -> "selectivity"
    text = strip_references(text)
    text = drop_junk_lines(text)
    return text
