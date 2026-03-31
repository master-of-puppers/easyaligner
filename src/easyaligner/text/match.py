from rapidfuzz.fuzz import partial_ratio_alignment

from easyaligner.data.datamodel import FuzzyMatch, SpeechSegment, WordSegment
from easyaligner.text.normalization import SpanMapNormalizer


def flatten_words(speeches: list[SpeechSegment]) -> list[WordSegment]:
    """Flatten all words from speech segments into a single list."""
    return [
        word for speech in speeches for alignment in speech.alignments for word in alignment.words
    ]


def build_haystack(words: list[WordSegment]) -> tuple[str, list[int]]:
    """Build a haystack string and char-to-word-index mapping from word segments.

    Concatenates word-level alignment texts and creates a mapping from each
    character index in the concatenated string to its source word index.

    Parameters
    ----------
    words : list[WordSegment]
        Word segments to concatenate.

    Returns
    -------
    haystack : str
        Concatenated text from the word segments.
    char_to_word : list[int]
        Mapping from character index in the concatenated string to word index.
    """
    char_to_word = []
    parts = []
    for i, word in enumerate(words):
        parts.append(word.text)
        char_to_word.extend([i] * len(word.text))
    return "".join(parts), char_to_word


def resolve_char_to_word(
    normalizer: SpanMapNormalizer,
    char_to_word: list[int],
) -> list[int]:
    """Compose a normalizer's span map with a char-to-word mapping.

    After normalizing a haystack with `SpanMapNormalizer`, this function creates
    a new char-to-word mapping for the normalized text by resolving each
    normalized character position back to a raw character position (via the
    span map), then to a word index.

    Parameters
    ----------
    normalizer : SpanMapNormalizer
        A normalizer that has been applied to the haystack text.
    char_to_word : list[int]
        Mapping from raw haystack character positions to word indices,
        as returned by `build_haystack`.

    Returns
    -------
    list[int]
        Mapping from normalized character positions to word indices.
    """
    normalized_char_to_word = []
    for span in normalizer.span_map:
        raw_pos = min(span[0], len(char_to_word) - 1)
        normalized_char_to_word.append(char_to_word[raw_pos])
    return normalized_char_to_word


def _fuzzy_match(
    needle: str,
    haystack: str,
    char_to_word: list[int],
    threshold: float = 55.0,
) -> FuzzyMatch | None:
    """Find a fuzzy match of needle text within a haystack string.

    Uses rapidfuzz's `partial_ratio_alignment` to locate the needle within the
    haystack, then maps character positions to word indices using the provided
    mapping.

    Parameters
    ----------
    needle : str
        The text to search for.
    haystack : str
        The text to search within (e.g. from `build_haystack`).
    char_to_word : list[int]
        Mapping from haystack character positions to word indices.
    threshold : float
        Minimum score (0-100) for a match to be returned.

    Returns
    -------
    FuzzyMatch or None
        The match result, or None if no match above the threshold.
    """
    if not haystack or not needle:
        return None

    alignment = partial_ratio_alignment(needle, haystack)

    if alignment.score < threshold:
        return None

    # Map character positions to word indices
    dest_start = alignment.dest_start
    dest_end = alignment.dest_end - 1  # partial_ratio_alignment end is exclusive

    # Clamp to valid range
    dest_start = max(0, min(dest_start, len(char_to_word) - 1))
    dest_end = max(0, min(dest_end, len(char_to_word) - 1))

    start_word_idx = char_to_word[dest_start]
    end_word_idx = char_to_word[dest_end]

    return FuzzyMatch(
        start_index=start_word_idx,
        end_index=end_word_idx,
        score=alignment.score,
    )


def _fuzzy_match_long(
    needle: str,
    haystack: str,
    char_to_word: list[int],
    threshold: float = 55.0,
    max_length: int = 300,
) -> FuzzyMatch | None:
    """Fuzzy match for potentially long needle texts.

    For very long needles (`> 2 * max_length` characters), splits the needle into
    two segments and matches them independently. The first segment is split from
    the start of the needle and the second segment from the end. `start_index`
    is determined by the start of the first segment's match, and `end_index` by the
    end of the second segment's match.

    Parameters
    ----------
    needle : str
        The (ground-truth) text to search for.
    haystack : str
        The (ASR) text to search within.
    char_to_word : list[int]
        Mapping from haystack character positions to word indices.
    threshold : float
        Minimum score (0-100) for a match to be returned.
    max_length : int
        Character length for splitting long needles.

    Returns
    -------
    FuzzyMatch or None
        The match result, or None if no match above the threshold.
    """
    if len(needle) <= 2 * max_length:
        return _fuzzy_match(needle, haystack, char_to_word, threshold)

    first_segment = needle[:max_length]
    last_segment = needle[-max_length:]

    match_first = _fuzzy_match(first_segment, haystack, char_to_word, threshold)
    match_last = _fuzzy_match(last_segment, haystack, char_to_word, threshold)

    if match_first is None or match_last is None:
        return None

    # Ensure ordering is correct
    if match_first.start_index > match_last.end_index:
        return None

    avg_score = (match_first.score + match_last.score) / 2

    return FuzzyMatch(
        start_index=match_first.start_index,
        end_index=match_last.end_index,
        score=avg_score,
    )


def fuzzy_match(
    needle: str,
    haystack: list[SpeechSegment],
    threshold: float = 55.0,
    max_length: int = 300,
    return_words: bool = False,
) -> FuzzyMatch | None | tuple[FuzzyMatch | None, list[WordSegment]]:
    """Fuzzy match between a needle (ground-truth text) and a haystack (ASR text).

    Flattens all word segments from the SpeechSegment objects into a single haystack and
    searches for the needle within them. The returned `FuzzyMatch` object includes both
    word indices and audio timestamps of the matched segment.

    Parameters
    ----------
    needle : str
        The text to search for.
    haystack : list[SpeechSegment]
        Speech segments containing word-level alignments to search within.
        The text from these segments will be concatenated to form the haystack.
    threshold : float
        Minimum score (0-100) for a match to be returned.
    max_length : int
        Character length for splitting long needles. Needles longer than
        ``2 * max_length`` are matched by anchoring the first and last
        ``max_length`` characters independently.
    return_words : bool
        If True, also return the flattened word list as a second value. Useful
        for debugging (e.g. inspecting surrounding context of the match).

    Returns
    -------
    FuzzyMatch or None or tuple[FuzzyMatch or None, list[WordSegment]]
        The match result with timestamps, or None if no match above the threshold.
        If `return_words` is True, returns a `(FuzzyMatch | None, list[WordSegment])`
        tuple instead, for debugging purposes.
    """
    words = flatten_words(haystack)
    haystack, char_to_word = build_haystack(words)
    result = _fuzzy_match_long(needle, haystack, char_to_word, threshold, max_length)

    if result is not None:
        result = FuzzyMatch(
            start_index=result.start_index,
            end_index=result.end_index,
            score=result.score,
            start=words[result.start_index].start,
            end=words[result.end_index].end,
        )

    if return_words:
        return result, words
    return result
