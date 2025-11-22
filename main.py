import character_extraction

TEXT_PATH = "./data/pride_and_prejudice_cleaned.txt"

def main():
    #character_extraction.ner_stanford(TEXT_PATH)
    character_extraction.ner_spacy(TEXT_PATH)
if __name__ == "__main__":
    main()
