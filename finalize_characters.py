#!/usr/bin/env python3
import os
import re
import json
from collections import defaultdict
from pathlib import Path

try:
    from rapidfuzz import fuzz
except Exception:
    raise SystemExit("Install rapidfuzz: pip install rapidfuzz")

try:
    import spacy
except Exception:
    raise SystemExit("Install spaCy and model en_core_web_sm: pip install spacy; python -m spacy download en_core_web_sm")

CLEANED_NAMES = Path("./results/character_extraction/cleaned_characters.txt")
SPACY_OCC = Path("./results/character_extraction/spacy_occurrences.txt")
STANFORD_OCC = Path("./results/character_extraction/stanford_occurrences.txt")
BOOK_TXT = Path("./data/pride_and_prejudice_cleaned.txt")
OUT_DIR = Path("./results/aliases")
FINAL_JSON = OUT_DIR / "final_characters.json"
INTERACTIONS_OUT = OUT_DIR / "interactions_by_sentence.txt"
SIM_THRESHOLD = 85

def load_cleaned_names(path):
    if not path.exists():
        raise FileNotFoundError(f"Cleaned names file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def parse_occurrences_file(path):
    if not path.exists():
        return {}
    occ = defaultdict(set)
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^====\s*(.+?)\s*====$", line)
            if m:
                current = m.group(1).strip()
                continue
            m2 = re.match(r"^\[(\d+)\]\s*(.*)$", line)
            if m2 and current:
                idx = int(m2.group(1))
                occ[current].add(idx)
    return {k: sorted(list(v)) for k, v in occ.items()}

def cluster_names_greedy(names, counts, threshold=SIM_THRESHOLD):
    mapping = {}
    groups = {}
    used = set()
    sorted_names = sorted(names, key=lambda n: (-counts.get(n,0), -len(n), n))
    for name in sorted_names:
        if name in used:
            continue
        canon = name
        group = [name]
        used.add(name)
        for other in names:
            if other in used:
                continue
            score = fuzz.token_sort_ratio(name, other)
            if score >= threshold or name.lower() in other.lower() or other.lower() in name.lower():
                group.append(other)
                used.add(other)
        groups[canon] = sorted(group)
        for g in group:
            mapping[g] = canon
    return mapping, groups

def find_missing_occurrences_by_scanning(book_text, names):
    nlp = spacy.load("en_core_web_sm")
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")
    doc = nlp(book_text)
    sentences = [sent.text for sent in doc.sents]
    try:
        from tqdm import tqdm
        iterator = enumerate(tqdm(sentences, desc="Scanning sentences"), start=1)
    except Exception:
        iterator = enumerate(sentences, start=1)
    occ = defaultdict(set)
    regexes = {name: re.compile(r"\b" + re.escape(name) + r"\b", flags=re.IGNORECASE) for name in names}
    for i, sent in iterator:
        for name, rx in regexes.items():
            if rx.search(sent):
                occ[name].add(i)
    return {k: sorted(list(v)) for k, v in occ.items()}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = load_cleaned_names(CLEANED_NAMES)
    if not names:
        raise SystemExit("No cleaned names found.")
    occ = parse_occurrences_file(SPACY_OCC)
    if not occ:
        occ = parse_occurrences_file(STANFORD_OCC)
    counts = {name: len(occ.get(name, [])) for name in names}
    mapping, groups = cluster_names_greedy(names, counts, threshold=SIM_THRESHOLD)
    missing = [n for n in names if n not in occ or not occ.get(n)]
    book_text = ""
    if missing:
        if not BOOK_TXT.exists():
            print("WARNING: Book file not found for scanning missing occurrences:", BOOK_TXT)
        else:
            print(f"Scanning book to find {len(missing)} missing names (this may take some time)...")
            with open(BOOK_TXT, "r", encoding="utf-8") as f:
                book_text = f.read()
            found_by_scan = find_missing_occurrences_by_scanning(book_text, missing)
            for k, v in found_by_scan.items():
                if v:
                    occ.setdefault(k, []).extend(v)
    final = {}
    for canon, aliases in groups.items():
        positions = set()
        for a in aliases:
            positions.update(occ.get(a, []))
        final[canon] = {
            "aliases": sorted(aliases),
            "positions": sorted(list(positions))
        }
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print("Saved final characters JSON to", FINAL_JSON)
    index_to_chars = defaultdict(list)
    for canon, info in final.items():
        for idx in info["positions"]:
            index_to_chars[idx].append(canon)
    interactions = []
    for idx in sorted(index_to_chars.keys()):
        chars = sorted(set(index_to_chars[idx]))
        if len(chars) >= 2:
            interactions.append((idx, chars))
    with open(INTERACTIONS_OUT, "w", encoding="utf-8") as f:
        for idx, chars in interactions:
            line = "(" + str(idx) + ", " + ", ".join(chars) + ")\n"
            f.write(line)
    print("Saved interactions (sentence-based) to", INTERACTIONS_OUT)
    print(f"Found {len(final)} canonical characters, {len(interactions)} sentence-level interactions.")

if __name__ == "__main__":
    main()