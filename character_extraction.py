#!/usr/bin/env python3
"""
Complete character extraction and alias resolution pipeline for Pride and Prejudice
Combines NER extraction, cleaning, alias grouping, and finalization into one workflow
"""
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# ============================================================================
# CONFIGURATION
# ============================================================================

TEXT_PATH = "./data/pride_and_prejudice_cleaned.txt"
RESULTS_DIR = Path("./results")
EXTRACTION_DIR = RESULTS_DIR / "character_extraction"
ALIASES_DIR = RESULTS_DIR / "aliases"

# Extraction outputs
STANFORD_CHARACTER_OUTPUT = EXTRACTION_DIR / "stanford_characters.txt"
STANFORD_OCCURRENCES_OUTPUT = EXTRACTION_DIR / "stanford_occurrences.txt"
SPACY_CHARACTER_OUTPUT = EXTRACTION_DIR / "spacy_characters.txt"
SPACY_OCCURRENCES_OUTPUT = EXTRACTION_DIR / "spacy_occurrences.txt"
GLINER_CHARACTER_OUTPUT = EXTRACTION_DIR / "gliner_characters.txt"
COMBINED_CHARACTER_OUTPUT = EXTRACTION_DIR / "combined_characters.txt"
CLEANED_FILE = EXTRACTION_DIR / "cleaned_characters.txt"

# Alias outputs
LEV_FILE = ALIASES_DIR / "aliases_levenshtein.json"
FUZZY_FILE = ALIASES_DIR / "aliases_fuzzy.json"
SEQ_FILE = ALIASES_DIR / "aliases_sequencematcher.json"

# Final output
FINAL_JSON = RESULTS_DIR / "final_characters.json"
INTERACTIONS_OUT = ALIASES_DIR / "interactions_by_sentence.txt"

# Parameters
SIMILARITY_THRESHOLD = 0.8
FINAL_SIM_THRESHOLD = 85

# Cleaning ignore list
IGNORE_WORDS = {
    "And", "Or", "The", "A", "An",
    "He", "She", "You", "I", "We", "They", "Him", "Them",
    "My", "Her", "His", "Father", "Mother", "Sister", "Uncle", "Papa",
    "Housekeeper", "Officer", "Contemptuously", "General", "Nay", "That",
    "Your", "Your Honoured", "Her Aunt", "Her Mother", "Her Ladyship",
    "Kitty And", "Don T", "Esq", "Michaelmas", "Chapter",
    "Netherfield", "Pemberley Woods", "Waiter", "Savours", "Scotch", "Sermons"
}

# ============================================================================
# STEP 1: NER EXTRACTION
# ============================================================================

def ner_stanford(text_path):
    """Extract characters using Stanford NER (Stanza)"""
    print("\n[1/7] Running Stanford NER...")
    try:
        import stanza
    except ImportError as e:
        raise SystemExit("Stanza not installed. Install with: pip install stanza") from e

    stanza.download('en', verbose=False)
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', device='cpu', verbose=False)

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    characters = set()
    occurrences = defaultdict(list)

    for i, sentence in enumerate(doc.sentences):
        for ent in sentence.ents:
            if ent.type == "PERSON":
                name = ent.text.strip()
                characters.add(name)
                occurrences[name].append((i, sentence.text.strip()))

    with open(STANFORD_CHARACTER_OUTPUT, "w", encoding="utf-8") as f:
        for name in sorted(characters):
            f.write(name + "\n")

    with open(STANFORD_OCCURRENCES_OUTPUT, "w", encoding="utf-8") as f:
        for name, occs in sorted(occurrences.items()):
            f.write(f"\n==== {name} ====\n")
            for idx, sent in occs:
                f.write(f"[{idx}] {sent}\n")

    print(f"   Found {len(characters)} unique characters with Stanford NER")
    return characters

def ner_spacy(text_path):
    """Extract characters using spaCy"""
    print("[2/7] Running spaCy NER...")
    import spacy
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError as e:
        raise SystemExit(
            "SpaCy model 'en_core_web_lg' not found.\n"
            "Install with: python -m spacy download en_core_web_lg"
        ) from e

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    characters = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])

    with open(SPACY_CHARACTER_OUTPUT, "w", encoding="utf-8") as f:
        for char in sorted(characters):
            f.write(char + "\n")

    sent_spans = list(doc.sents)
    occurrences = defaultdict(list)

    for i, s in enumerate(sent_spans):
        for ent in s.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                occurrences[name].append((i, s.text.strip()))

    with open(SPACY_OCCURRENCES_OUTPUT, "w", encoding="utf-8") as f:
        for name, occs in sorted(occurrences.items()):
            f.write(f"\n==== {name} ====\n")    
            for idx, sent in occs:
                f.write(f"[{idx}] {sent}\n")

    print(f"   Found {len(characters)} unique characters with spaCy")
    return characters

def ner_gliner(text_path):
    """Extract characters using GLiNER"""
    print("[3/7] Running GLiNER...")
    try:
        from gliner import GLiNER
    except ImportError as e:
        raise SystemExit("GLiNER not installed. Install with: pip install gliner") from e
    
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    labels = ["person"]
    chunk_size = 1500
    all_entities = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        entities = model.predict_entities(chunk, labels, threshold=0.5)
        all_entities.extend(entities)

    counter = Counter((e["text"], e["label"]) for e in all_entities)

    with open(GLINER_CHARACTER_OUTPUT, "w", encoding="utf-8") as f:
        for (txt, lbl), cnt in counter.most_common():
            f.write(f"{txt}\t{lbl}\t{cnt}\n")

    print(f"   Found {len(counter)} unique characters with GLiNER")
    return set(txt for txt, _ in counter.keys())

def combine_ner_results(stanford_file, spacy_file, gliner_file, output_file):
    """Combine results from all NER methods"""
    sets = []
    for file_path in [stanford_file, spacy_file, gliner_file]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t")[0] for line in f if line.strip()]
            sets.append(set(lines))

    combined_characters = set().union(*sets)

    with open(output_file, "w", encoding="utf-8") as f:
        for char in sorted(combined_characters):
            f.write(char + "\n")

    print(f"   Combined: {len(combined_characters)} unique characters")
    return combined_characters

# ============================================================================
# STEP 2: CLEANING
# ============================================================================

def normalize_name(name: str) -> str:
    """Normalize character name"""
    name = re.sub(r"[''""\"!?,;:_]", " ", name)
    name = re.sub(r"--.*$", "", name)
    name = re.sub(r"\b[A-Z]\.\b", "", name)
    name = re.sub(r"\d+", "", name)
    name = " ".join(name.split())
    return name.title()

def clean_names(names):
    """Clean and filter character names"""
    print("[4/7] Cleaning character names...")
    cleaned = []
    for n in names:
        norm = normalize_name(n)
        if len(norm) < 2:
            continue

        words = norm.split()
        if not words:
            continue

        if any(word in IGNORE_WORDS or len(word) == 1 for word in words):
            continue

        cleaned.append(" ".join(words))
    
    cleaned = sorted(set(cleaned))
    print(f"   Cleaned: {len(cleaned)} names (removed {len(names) - len(cleaned)} invalid entries)")
    return cleaned

# ============================================================================
# STEP 3: ALIAS RESOLUTION (Multiple methods)
# ============================================================================

def levenshtein(a, b):
    """Calculate Levenshtein distance"""
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
    """Levenshtein-based similarity ratio"""
    distance = levenshtein(a.lower(), b.lower())
    max_len = max(len(a), len(b))
    return 1 - distance / max_len if max_len else 1.0

def seq_similarity(a, b):
    """SequenceMatcher similarity ratio"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def fuzzy_similarity(a, b):
    """FuzzyWuzzy similarity ratio"""
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        from rapidfuzz import fuzz
    return fuzz.ratio(a.lower(), b.lower()) / 100

def build_aliases(names, similarity_func, threshold=SIMILARITY_THRESHOLD):
    """Build alias groups using similarity function"""
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

def generate_alias_files(names):
    """Generate alias files using different similarity methods"""
    print("[5/7] Generating alias groups...")
    
    lev_aliases = build_aliases(names, lev_similarity)
    with open(LEV_FILE, "w", encoding="utf-8") as f:
        json.dump(lev_aliases, f, ensure_ascii=False, indent=2)
    print(f"   Levenshtein: {len(lev_aliases)} groups → {LEV_FILE}")

    fuzzy_aliases = build_aliases(names, fuzzy_similarity)
    with open(FUZZY_FILE, "w", encoding="utf-8") as f:
        json.dump(fuzzy_aliases, f, ensure_ascii=False, indent=2)
    print(f"   FuzzyWuzzy: {len(fuzzy_aliases)} groups → {FUZZY_FILE}")

    seq_aliases = build_aliases(names, seq_similarity)
    with open(SEQ_FILE, "w", encoding="utf-8") as f:
        json.dump(seq_aliases, f, ensure_ascii=False, indent=2)
    print(f"   SequenceMatcher: {len(seq_aliases)} groups → {SEQ_FILE}")

# ============================================================================
# STEP 4: FINALIZATION
# ============================================================================

def parse_occurrences_file(path):
    """Parse occurrence file to get sentence indices"""
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

def cluster_names_greedy(names, counts, threshold=FINAL_SIM_THRESHOLD):
    """
    Cluster character names using a greedy algorithm,
    where similarity is calculated as the average of three metrics:
    Levenshtein, FuzzyWuzzy, and SequenceMatcher
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        from fuzzywuzzy import fuzz
        
    mapping = {}
    groups = {}
    used = set()
    sorted_names = sorted(names, key=lambda n: (-counts.get(n, 0), -len(n), n))
    
    for name in sorted_names:
        if name in used:
            continue
        canon = name
        group = [name]
        used.add(name)
        
        for other in names:
            if other in used:
                continue
            # Compute average similarity across three metrics
            # print(lev_similarity(name, other), fuzzy_similarity(name, other), seq_similarity(name, other))
            avg_similarity = (
                                     lev_similarity(name, other) +
                                     fuzzy_similarity(name, other) +
                                     seq_similarity(name, other)
                             ) / 3
            # Add to group if average similarity is above threshold or if one name is contained in the other
            if avg_similarity >= threshold or name.lower() in other.lower() or other.lower() in name.lower():
                group.append(other)
                used.add(other)
        
        groups[canon] = sorted(group)
        for g in group:
            mapping[g] = canon
    
    return mapping, groups

def find_missing_occurrences_by_scanning(book_text, names):
    """Scan book text to find occurrences of missing names"""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")
    
    doc = nlp(book_text)
    sentences = [sent.text for sent in doc.sents]
    
    try:
        from tqdm import tqdm
        iterator = enumerate(tqdm(sentences, desc="   Scanning sentences"), start=1)
    except Exception:
        iterator = enumerate(sentences, start=1)
    
    occ = defaultdict(set)
    regexes = {name: re.compile(r"\b" + re.escape(name) + r"\b", flags=re.IGNORECASE) for name in names}
    
    for i, sent in iterator:
        for name, rx in regexes.items():
            if rx.search(sent):
                occ[name].add(i)
    
    return {k: sorted(list(v)) for k, v in occ.items()}

def finalize_characters(cleaned_names):
    """Create final character JSON with aliases and positions"""
    print("[6/7] Finalizing character data...")
    
    # Load occurrences
    occ = parse_occurrences_file(SPACY_OCCURRENCES_OUTPUT)
    if not occ:
        occ = parse_occurrences_file(STANFORD_OCCURRENCES_OUTPUT)
    
    # Calculate counts
    counts = {name: len(occ.get(name, [])) for name in cleaned_names}
    
    # Cluster names
    mapping, groups = cluster_names_greedy(cleaned_names, counts, threshold=FINAL_SIM_THRESHOLD)
    
    # Find missing occurrences
    missing = [n for n in cleaned_names if n not in occ or not occ.get(n)]
    
    if missing and Path(TEXT_PATH).exists():
        print(f"   Scanning book for {len(missing)} missing names...")
        with open(TEXT_PATH, "r", encoding="utf-8") as f:
            book_text = f.read()
        found_by_scan = find_missing_occurrences_by_scanning(book_text, missing)
        for k, v in found_by_scan.items():
            if v:
                occ.setdefault(k, []).extend(v)
    
    # Build final structure
    final = {}
    for canon, aliases in groups.items():
        positions = set()
        for a in aliases:
            positions.update(occ.get(a, []))
        final[canon] = {
            "aliases": sorted(aliases),
            "positions": sorted(list(positions))
        }
    
    # Save final JSON
    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    
    print(f"   Saved {len(final)} canonical characters → {FINAL_JSON}")
    
    # Generate interactions
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
    
    print(f"   Saved {len(interactions)} sentence-level interactions → {INTERACTIONS_OUT}")
    
    return final

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete character extraction and processing pipeline"""
    print("="*70)
    print("CHARACTER EXTRACTION & ALIAS RESOLUTION PIPELINE")
    print("="*70)
    
    # Create output directories
    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    ALIASES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1-3: Extract characters using all NER methods
    ner_stanford(TEXT_PATH)
    ner_spacy(TEXT_PATH)
    ner_gliner(TEXT_PATH)
    
    # Combine results
    combined = combine_ner_results(
        STANFORD_CHARACTER_OUTPUT,
        SPACY_CHARACTER_OUTPUT,
        GLINER_CHARACTER_OUTPUT,
        COMBINED_CHARACTER_OUTPUT
    )
    
    # Step 4: Clean names
    with open(COMBINED_CHARACTER_OUTPUT, "r", encoding="utf-8") as f:
        raw_names = [line.strip() for line in f if line.strip()]
    
    cleaned = clean_names(raw_names)
    
    with open(CLEANED_FILE, "w", encoding="utf-8") as f:
        for name in cleaned:
            f.write(name + "\n")
    
    # Step 5: Generate alias files
    generate_alias_files(cleaned)
    
    # Step 6: Finalize
    final = finalize_characters(cleaned)
    
    print("\n" + "="*70)
    print("[7/7] PIPELINE COMPLETE!")
    print("="*70)
    print(f"Extracted and cleaned {len(cleaned)} character names")
    print(f"Created {len(final)} canonical character groups")
    print(f"Final output: {FINAL_JSON}")
    print("="*70)

if __name__ == "__main__":
    main()
