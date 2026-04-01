import re

import nltk
from nltk.tokenize.punkt import PunktTokenizer


def load_tokenizer(language: str = "swedish") -> PunktTokenizer:
    """
    Loads a PunktTokenizer for the specified language that can be used to sentence tokenize text.

    Parameters
    ----------
    language : str, default "swedish"
        Language to use for the tokenizer, e.g. "swedish", "english".

    Returns
    -------
    PunktTokenizer
        Loaded tokenizer.
    """
    try:
        tokenizer = PunktTokenizer(lang=language)
    except LookupError:
        nltk.download("punkt_tab")
        tokenizer = PunktTokenizer(lang=language)

    return tokenizer


def paragraph_tokenizer(text: str) -> list[tuple[int, int]]:
    """
    Tokenize text into paragraphs based on double newlines.

    Returns character spans (start, end) for each non-empty paragraph.
    Suitable as a drop-in replacement for PunktTokenizer when paragraph-level
    alignment granularity is desired.

    Parameters
    ----------
    text : str
        The text to tokenize into paragraphs.

    Returns
    -------
    list of tuple[int, int]
        List of (start_char, end_char) spans, one per paragraph.
    """
    spans = []
    start = 0
    for m in re.finditer(r'\r?\n\r?\n', text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]
