from collections import defaultdict

def ner_stanford(text_path):
    try:
        import stanza
    except ImportError as e:
        raise SystemExit(
            "Stanza isn't installed. Install with: pip install stanza"
        ) from e

    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

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

    with open("./results/stanford_characters.txt", "w", encoding="utf-8") as f:
        for name in sorted(characters):
            f.write(name + "\n")

    with open("./results/stanford_occurrences.txt", "w", encoding="utf-8") as f:
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

    doc = nlp(text)

    # Extract PERSON entities
    characters = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    #print(f"Total unique PERSON entities: {len(characters)}\n {list(characters)}")

    # Save characters to a file
    with open("./results/spacy_characters.txt", "w", encoding="utf-8") as f:
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
    with open("./results/spacy_occurrences.txt", "w", encoding="utf-8") as f:
        for name, occs in sorted(occurrences.items()):
            f.write(f"\n==== {name} ====\n")    
            for idx, sent in occs:
                f.write(f"[{idx}] {sent}\n")


if __name__ == "__main__":
    text_path = "./data/pride_and_prejudice_cleaned.txt"
    ner_stanford(text_path)
    ner_spacy(text_path)