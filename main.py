import character_extraction

def main():
    #character_extraction.ner_stanford("./data/pride_and_prejudice.txt")
    character_extraction.ner_spacy("./data/pride_and_prejudice.txt")

if __name__ == "__main__":
    main()
