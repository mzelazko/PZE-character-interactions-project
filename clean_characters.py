import re

INPUT_FILE = "./results/combined_characters.txt"
CLEANED_FILE = "./results/cleaned_characters.txt"

IGNORE_WORDS = {
    "And", "Or", "The", "A", "An",
    "He", "She", "You", "I", "We", "They", "Him", "Them",
    "My", "Her", "His", "Father", "Mother", "Sister", "Uncle", "Papa",
    "Housekeeper", "Officer", "Contemptuously", "General", "Nay", "That",
    "Your", "Your Honoured", "Her Aunt", "Her Mother", "Her Ladyship",
    "Kitty And", "Don T", "Esq", "Michaelmas", "Chapter",
    "Netherfield", "Pemberley Woods", "Waiter", "Savours", "Scotch", "Sermons"
}

def load_raw_names(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def normalize_name(name: str) -> str:
    name = re.sub(r"[’‘”“\"!?,;:_]", " ", name)
    name = re.sub(r"--.*$", "", name)
    name = re.sub(r"\b[A-Z]\.\b", "", name)
    name = re.sub(r"\d+", "", name)
    name = " ".join(name.split())
    return name.title()

def clean_names(names):
    cleaned = []
    for n in names:
        norm = normalize_name(n)
        if len(norm) < 2:
            continue

        words = norm.split()

        if not words: # Mr
            continue

        if any(word in IGNORE_WORDS or len(word) == 1 for word in words):
            continue

        cleaned.append(" ".join(words))
    return sorted(set(cleaned))


if __name__ == "__main__":
    raw = load_raw_names(INPUT_FILE)
    cleaned = clean_names(raw)
    with open(CLEANED_FILE, "w", encoding="utf-8") as f:
        for name in cleaned:
            f.write(name + "\n")
