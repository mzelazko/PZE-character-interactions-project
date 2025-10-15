import stanza
from collections import defaultdict

def main():
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    with open("./data/pride_and_prejudice.txt", "r", encoding="utf-8") as f:
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

if __name__ == "__main__":
    main()