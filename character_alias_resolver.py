import json
from pathlib import Path
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz

CLEANED_FILE = Path("./results/character_extraction/cleaned_characters.txt")
OUTPUT_DIR = Path("./results/aliases")
LEV_FILE = OUTPUT_DIR / "aliases_levenshtein.json"
FUZZY_FILE = OUTPUT_DIR / "aliases_fuzzy.json"
SEQ_FILE = OUTPUT_DIR / "aliases_sequencematcher.json"

# SIMILARITY_THRESHOLD = 0.75
SIMILARITY_THRESHOLD = 0.8

def load_names(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def levenshtein(a, b):
    # https://pl.wikipedia.org/wiki/Odleg%C5%82o%C5%9B%C4%87_Levenshteina
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current_row = list(range(n + 1))
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (a[j - 1] != b[i - 1])
            current_row[j] = min(insertions, deletions, substitutions)
    return current_row[n]

def lev_similarity(a, b):
    distance = levenshtein(a.lower(), b.lower())
    max_len = max(len(a), len(b))
    return 1 - distance / max_len if max_len else 1.0

def seq_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def fuzzy_similarity(a, b):
    # https://medium.com/@bravekjh/unlocking-the-power-of-fuzzy-matching-in-python-a-practical-guide-ec37ebd8f3eb
    return fuzz.ratio(a.lower(), b.lower()) / 100

def build_aliases(names, similarity_func, threshold=SIMILARITY_THRESHOLD):
    groups = []
    used = set()
    for name in names:
        if name in used:
            continue
        group = [name]
        used.add(name)
        for other in names:
            if other in used:
                continue
            if similarity_func(name, other) >= threshold:
                group.append(other)
                used.add(other)
        groups.append(sorted(group))
    return {group[0]: group for group in groups}

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = load_names(CLEANED_FILE)

    # Levenshtin
    lev_aliases = build_aliases(names, lev_similarity)
    with open(LEV_FILE, "w", encoding="utf-8") as f:
        json.dump(lev_aliases, f, ensure_ascii=False, indent=2)
    print(f"Levenshtein:{LEV_FILE}")

    # FuzzyWuzzy
    fuzzy_aliases = build_aliases(names, fuzzy_similarity)
    with open(FUZZY_FILE, "w", encoding="utf-8") as f:
        json.dump(fuzzy_aliases, f, ensure_ascii=False, indent=2)
    print(f"FuzzyWuzzy : {FUZZY_FILE}")

    # SequenceMatcher
    seq_aliases = build_aliases(names, seq_similarity)
    with open(SEQ_FILE, "w", encoding="utf-8") as f:
        json.dump(seq_aliases, f, ensure_ascii=False, indent=2)
    print(f"SequenceMatcher: {SEQ_FILE}")
