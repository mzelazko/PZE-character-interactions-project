<p align="center">
  <img src="book-cover.jpg" width="300"/>
</p>

# Interactions of Characters in Pride and Prejudice Novel

*Project for the course "Projekt Zespołowy (II Stopień)" at UKSW.*


## Team 5

Team Members:
- Michał Żelazko, Manager (planning, research on SOTA methods, implementation of book processing and NER tools, repo management, validating others work, presenting every week progress on the project to the professor)
- Norbert Zabost, Documentation
- Polina Konecka
- Damian Kantorowski
- Rafał Wielądek

## Tech Stack
| Layer | Technologies |
|--------|----------------|
| **Language** | Python 3 |
| **NER (character extraction)** | [spaCy](https://spacy.io/) (`en_core_web_lg`), [Stanza](https://stanfordnlp.github.io/stanza/) (English NER), [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_medium-v2.1`) |
| **Coreference** | [fastcoref](https://github.com/shon-otmazgin/fastcoref) |
| **Models & pipelines** | [Hugging Face Transformers](https://huggingface.co/docs/transformers)|
| **Local LLM** | [Ollama](https://ollama.com/)|
| **Alias matching & clustering** | [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) / [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz), `difflib.SequenceMatcher`, custom Levenshtein-style similarity, greedy clustering |
| **Utilities** | `tqdm`, `requests` |

## Current State of The Project

Detailed documentation is available in the [report](./report.pdf).

### Completed Tasks
- Downloaded and cleaned the book for processing ([process_book.py](./process_book.py) and [data](./data))
- Researched related work to identify state-of-the-art (SOTA) methods (see highlights in [papers](./papers/))
- Tested three Named Entity Recognition (NER) models on the book ([character_extraction.py](./character_extraction.py) and [results/character_extraction](./results/character_extraction/)):
  - `en_core_web_lg` from the `Spacy` libarry
  - stanford NER
  - [GLiNER](https://github.com/urchade/GLiNER)
- Combined outputs from all models to maximize recall
- Implemented and compared several string-based similarity methods for alias grouping:
  - Levenshtein-based similarity (custom implementation),
  - SequenceMatcher (difflib),
  - Fuzzy string matching (FuzzyWuzzy / RapidFuzz).
- Applied a final greedy clustering algorithm, which:
  - sorts names by frequency,
  - iteratively groups the closest variants under a canonical name,
  - uses the average similarity score computed from Levenshtein, fuzzy matching, and SequenceMatcher, 
  - merges names when one variant is contained within another  [final_characters](./results/final_characters.json)).
- Interaction detection ([interactions.py](./interactions.py)) using five methods:
  - Method 1: Sliding window + pronoun heuristics - co-occurrence in the same line or close context + simple heuristic pronoun resolution using gender and recency.
  - Method 2 (fastcoref): Generates clusters for each text chunk. If a cluster contains a name, all words in that cluster are mapped to that character. Co-occurrence of different character clusters = interaction.
  - Method 3 (Contextual Weighting): Uses decaying weights and Sentiment Analysis (TextBlob) with regex-based Speaker Detection.
  - Method 4 (Local LLM analysis (Ollama)): Prompts an LLM with the text chunk and asks it to respond with a list of interactions.
  - Method 5 (Advanced Contextual): Adds Zero-Shot Classification (BART) for deep dialogue  attribution

## Research

English and Polish Public domain books are available at https://www.gutenberg.org/ and https://wolnelektury.pl/.

We have lecture presentations from the Natural Language Processing course, kindly shared with us by dr. Katsiaryna Kosarava.

### Related work
- A comprehensive overview of the methods used for creating character networks: [Extraction and Analysis of Fictional Character Networks](papers/Extraction%20and%20Analysis%20of%20Fictional%20Character%20Networks.pdf)
- Character network analysis based on "A Song of Ice and Fire" novels and "Game of Thrones" series: https://networkofthrones.com/
- Examples of character network visualizations on movies: https://moviegalaxies.com/discover/movies/all/