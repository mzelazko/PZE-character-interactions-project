import re
from functools import lru_cache
from typing import NamedTuple

from character_pipeline.config import NAME_PARTICLES, NICKNAMES, TITLES


class ParsedName(NamedTuple):
    title: str
    given: tuple[str, ...]
    surname: str


def strip_punct(tok: str) -> str:
    return re.sub(r"[.,;:!?]", "", tok).strip()


@lru_cache(maxsize=1)
def ignore_words() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords

        try:
            sw = set(stopwords.words("english"))
        except LookupError:
            import nltk

            nltk.download("stopwords", quiet=True)
            sw = set(stopwords.words("english"))
    except Exception:
        sw = {
            "the", "a", "an", "and", "or", "he", "she", "i", "you", "we", "they", "it",
            "my", "your", "his", "her", "its", "our", "their", "that", "this", "nay",
        }
    sw |= {
        "father", "mother", "sister", "brother", "uncle", "aunt", "papa", "mama", "mamma",
        "son", "daughter", "niece", "nephew", "cousin", "husband", "wife", "child",
        "ladyship", "lordship", "honour", "honoured", "honor",
        "housekeeper", "gardener", "apothecary", "officer", "waiter", "servant", "man", "woman",
        "mr", "mrs", "ms", "miss", "sir", "lady", "lord", "dr", "rev", "esq", "esquire",
        "captain", "colonel", "major", "general", "prince", "princess", "king", "queen",
        "duke", "duchess",
    }
    return frozenset(sw)


def parse_name(name: str) -> ParsedName:
    """Parse a name into title, given names, and surname."""
    tokens = name.split()
    if len(tokens) >= 3 and tokens[-2].lower() == "of":
        tokens = tokens[:-2]
    title = ""
    i = 0
    while i < len(tokens) and strip_punct(tokens[i]).lower() in TITLES:
        title = strip_punct(tokens[i]).lower()
        i += 1
    rest = tokens[i:]
    while rest and strip_punct(rest[-1]).lower() in NAME_PARTICLES:
        rest = rest[:-1]
    if not rest:
        return ParsedName(title, (), "")
    sur_toks = [rest[-1]]
    j = len(rest) - 2
    while j >= 0 and strip_punct(rest[j]).lower() in NAME_PARTICLES:
        sur_toks.insert(0, rest[j])
        j -= 1
    given = tuple(
        NICKNAMES.get(strip_punct(g).lower(), strip_punct(g).lower())
        for g in rest[: j + 1]
        if strip_punct(g)
    )
    surname = strip_punct(" ".join(sur_toks)).lower()
    return ParsedName(title, given, surname)
