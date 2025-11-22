from collections import defaultdict

TEXT_PATH = "./data/pride_and_prejudice_cleaned.txt"
STANFORD_CHARACTER_OUTPUT = "./results/character_extraction/stanford_characters.txt"
STANFORD_OCCURRENCES_OUTPUT = "./results/character_extraction/stanford_occurrences.txt"
SPACY_CHARACTER_OUTPUT = "./results/character_extraction/spacy_characters.txt"
SPACY_OCCURRENCES_OUTPUT = "./results/character_extraction/spacy_occurrences.txt"
GLINER_CHARACTER_OUTPUT = "./results/character_extraction/gliner_characters.txt"
#GLINER_OCCURRENCES_OUTPUT = "./results/character_extraction/gliner_occurrences.txt"
COMBINED_CHARACTER_OUTPUT = "./results/character_extraction/combined_characters.txt"

def ner_stanford(text_path):
    try:
        import stanza
    except ImportError as e:
        raise SystemExit(
            "Stanza isn't installed. Install with: pip install stanza"
        ) from e

    stanza.download('en')
    nlp = stanza.Pipeline(
        lang='en',
        processors='tokenize,ner',
        device='cpu' # for newer GPUs set to 'cuda'
        )

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # text = text[:20000]
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

    # print(f'Done. Found {len(characters)} unique PERSON entities')

def ner_spacy(text_path):
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

    # text = text[:20000]

    doc = nlp(text)
    

    # Extract PERSON entities
    characters = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    #print(f"Total unique PERSON entities: {len(characters)}\n {list(characters)}")

    # Save characters to a file
    with open(SPACY_CHARACTER_OUTPUT, "w", encoding="utf-8") as f:
        for char in sorted(characters):
            f.write(char + "\n")

    # Split into sentences
    sent_spans = list(doc.sents)

    occurrences = defaultdict(list)

    # Build occurrences in sentence order
    for i, s in enumerate(sent_spans):
        for ent in s.ents:
            if ent.label_ == "PERSON":
                #name = " ".join(ent.text.split())
                name = ent.text.strip()
                occurrences[name].append((i, s.text.strip()))

    # Save occurrences to a file
    with open(SPACY_OCCURRENCES_OUTPUT, "w", encoding="utf-8") as f:
        for name, occs in sorted(occurrences.items()):
            f.write(f"\n==== {name} ====\n")    
            for idx, sent in occs:
                f.write(f"[{idx}] {sent}\n")

#
def ner_GLiNER2(text_path):
    try:
        from gliner import GLiNER
    except ImportError as e:
        raise SystemExit(
            "GLiNER isn't installed. Install with: pip install gliner2"
        ) from e
    
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    #text = text[:20000]

    # Most GLiNER models should work best when entity types are in lower case or title case
    labels = ["person"]

    chunk_size = 1500  # model takes max 384 tokens
    all_entities = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        entities = model.predict_entities(chunk, labels, threshold=0.5)
        all_entities.extend(entities)

    from collections import Counter

    counter = Counter((e["text"], e["label"]) for e in all_entities)
    output_path = GLINER_CHARACTER_OUTPUT

    with open(output_path, "w", encoding="utf-8") as f:
            for (txt, lbl), cnt in counter.most_common():
                f.write(f"{txt}\t{lbl}\t{cnt}\n")

    print(f"Saved {len(counter)} unique entities to {output_path}")

def combine_ner_results(stanford_file, spacy_file, gliner_file, output_file):
    sets = []
    for file_path in [stanford_file, spacy_file, gliner_file]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t")[0] for line in f if line.strip()]
            sets.append(set(lines))

    comined_characters = set().union(*sets)

    with open(output_file, "w", encoding="utf-8") as f:
        for char in sorted(comined_characters):
            f.write(char + "\n")

    print(f"Saved {len(comined_characters)} unique characters to {output_file}")


if __name__ == "__main__":
    ner_stanford(TEXT_PATH)
    ner_spacy(TEXT_PATH)
    ner_GLiNER2(TEXT_PATH)

    combine_ner_results(
        STANFORD_CHARACTER_OUTPUT,
        SPACY_CHARACTER_OUTPUT,
        GLINER_CHARACTER_OUTPUT,
        COMBINED_CHARACTER_OUTPUT
    )
