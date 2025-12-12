<p align="center">
  <img src="book-cover.jpg" width="300"/>
</p>

# Interactions of Characters in Pride and Prejudice Novel

*Project for the course "Projekt Zespołowy (II Stopień)" at UKSW.*


## Team 5

Team Members:
- Michał Żelazko, Manager
- Norbert Zabost, Documentation
- Polina Konetskaia, Github repo
- Damian Kantorowski
- Rafał Wielądek


## Current State of The Project

### What is done
- Downloaded and cleaned the book for processing ([process_book.py](./process_book.py) and [data](./data))
- Researched related work to identify state-of-the-art (SOTA) methods (see highlights in [papers](./papers/))
- Tested three Named Entity Recognition (NER) models on the book ([character_extraction.py](./character_extraction.py) and [results/character_extraction](./results/character_extraction/)):
  - `en_core_web_lg` from the `Spacy` libarry
  - stanford NER
  - [GLiNER](https://github.com/urchade/GLiNER)
- Performed initial alias resolution ([character_alias_resolver.py](./character_alias_resolver.py) and [alias results](./results/aliases)) using three different methods:
  - Levenshtein distance  
    - one implementation written from scratch  
    - one using the `fuzz` function from the `fuzzywuzzy` library with additional features
  - Gestalt pattern matching using the `SequenceMatcher` class from the [difflib](https://docs.python.org/3/library/difflib.html) library
- Interaction detection ([interactions.py](./interactions.py)) using three methods:
  - Method 1: Pronoun-based matching. Finds the first character matching the gender. For plural pronouns, it picks the last 2 characters.
  - Method 2 (fastcoref): Generates clusters for each text chunk. If a cluster contains a name, all words in that cluster are mapped to that character. Co-occurrence of different character clusters = interaction.
  - Method 3 (LLM): Prompts an LLM with the text chunk and character list and asks it to respond with a list of interactions.


### Tasks
- Gender mapping in first method needs refinement once the character list is finalized.
- Use a better soloution for chunking to avoid splitting characters in the same sentence across chunks.
- Perform tests using different LLMs and various text chunk sizes.

## Research

English and Polish Public domain books are available at https://www.gutenberg.org/ and https://wolnelektury.pl/.

We have lecture presentations from the Natural Language Processing course, kindly shared with us by dr. Katsiaryna Kosarava.

### Related work
- A comprehensive overview of the methods used for creating character networks: [Extraction and Analysis of Fictional Character Networks](papers/Extraction%20and%20Analysis%20of%20Fictional%20Character%20Networks.pdf)
- Character network analysis based on "A Song of Ice and Fire" novels and "Game of Thrones" series: https://networkofthrones.com/
- Examples of character network visualizations on movies: https://moviegalaxies.com/discover/movies/all/



## Notes

- Prepare a few short text samples that highlight specific challenges the method may encounter and test how well it handles them. For example:

  > A and B were discussing in the kitchen. C enters the room and says something to them.

  > A and B were discussing in the kitchen, while D and E were in the garden. C enters the kitchen and says something.
