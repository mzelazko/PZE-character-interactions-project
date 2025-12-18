import character_extraction
import interactions

def main():
    #character_extraction.ner_stanford("./data/pride_and_prejudice.txt")
	
# Set the flag to enable coreference resolution
    USE_COREF = True

    occurrences = character_extraction.ner_spacy(
        text_path="./data/pride_and_prejudice.txt",
        use_coref=USE_COREF
    )
    
    # Pass the occurrences to the interactions module
    interactions.main(occurrences)

if __name__ == "__main__":
    main()
