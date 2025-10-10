import re
import spacy
import en_core_web_lg # python -m spacy download en_core_web_lg
from collections import defaultdict

def strip_gutenberg_headers(text: str) -> str:

    start_marker = "*** START OF THE PROJECT GUTENBERG"
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    
    start_idx = text.find(start_marker)
    if start_idx != -1:
        # Move to the line after the start marker
        text = text[start_idx + len(start_marker):]
        # Cut from the next newline to remove the remaining part of the marker line
        newline_idx = text.find('\n')
        if newline_idx != -1:
            text = text[newline_idx + 1:]
    
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]
    
    return text.strip()

def main():
    nlp = en_core_web_lg.load()

    with open("./data/pride_and_prejudice.txt", "r", encoding="utf-8") as f:
        text = f.read()
    text = strip_gutenberg_headers(text)

    doc = nlp(text)

    # Extract PERSON entities
    characters = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    #print(f"Total unique PERSON entities: {len(characters)}\n {list(characters)}")

    # Save characters to a file
    with open("./results/characters.txt", "w", encoding="utf-8") as f:
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
    with open("./results/character_occurrences.txt", "w", encoding="utf-8") as f:
        for name, occs in sorted(occurrences.items()):
            f.write(f"\n==== {name} ====\n")    
            for idx, sent in occs:
                f.write(f"[{idx}] {sent}\n")

if __name__ == "__main__":
    main()
