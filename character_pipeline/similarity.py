from difflib import SequenceMatcher


def seq_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fuzzy_similarity(a: str, b: str) -> float:
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        from rapidfuzz import fuzz
    return fuzz.ratio(a.lower(), b.lower()) / 100


def seq_fuzz_similarity(a: str, b: str) -> float:
    return (seq_similarity(a, b) + fuzzy_similarity(a, b)) / 2
