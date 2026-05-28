import re

from character_pipeline.config import CHAPTER_RE, NAME_PARTICLES, PLACE_NAMES
from character_pipeline.name_parsing import ignore_words


def normalize_name(name: str) -> str:
    name = re.sub(r"['’]s?$|s['’]$", "", name)
    name = re.sub(r"\s*(?:,--|!--|\?--|--).*$", "", name)
    name = re.sub(r"[!?,;:_]", " ", name)
    name = " ".join(name.split())
    if CHAPTER_RE.match(name):
        return ""
    if name and name == name.upper():
        name = " ".join(
            w.lower() if w.lower() in NAME_PARTICLES else w.capitalize()
            for w in name.split()
        )
    return name


def clean_names(names: list[str]) -> list[str]:
    ignore = ignore_words()
    cleaned: dict[str, str] = {}
    for n in names:
        if not n:
            continue
        norm = normalize_name(n)
        if len(norm) < 2 or any(c.isdigit() for c in norm):
            continue
        words = norm.split()
        if not words or any(len(w) == 1 for w in words):
            continue
        if not all(w[0].isupper() or w.lower() in NAME_PARTICLES for w in words):
            continue
        if not any(w[0].isupper() and w.lower() not in NAME_PARTICLES for w in words):
            continue
        if all(w.lower().rstrip(".") in ignore for w in words):
            continue
        if norm.lower() in PLACE_NAMES:
            continue
        key = norm.lower()
        if key not in cleaned or (cleaned[key].isupper() and not norm.isupper()):
            cleaned[key] = norm
    return sorted(cleaned.values())
