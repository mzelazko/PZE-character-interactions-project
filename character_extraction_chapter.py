import torch
import re
import spacy
from collections import defaultdict
from fastcoref import FCoref

def strip_gutenberg_headers(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG"
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    
    start_idx = text.find(start_marker)
    if start_idx != -1:
        text = text[start_idx + len(start_marker):]
        newline_idx = text.find('\n')
        if newline_idx != -1:
            text = text[newline_idx + 1:]
    
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]
    
    return text.strip()

def get_first_chapter(text):
    """
    Extracts the text up to the start of the second chapter.
    This is a heuristic for Pride and Prejudice.
    """
    # Find the end of Chapter 1, which is typically marked by "CHAPTER II" or similar.
    # We use a regex to be slightly more robust.
    match = re.search(r'CHAPTER\s+(II|2)', text, re.IGNORECASE)
    if match:
        return text[:match.start()].strip()
    
    # Fallback: return the first 10,000 characters if chapter marker is not found
    return text[:10000].strip()

def resolve_coreferences(text):
    """
    Performs coreference resolution on the text using the lightweight fastcoref library.
    Returns the text with anaphors replaced by their antecedents.
    """
    # Use the get_first_chapter function to limit the text size
    text_fragment = get_first_chapter(text)
    
    try:
        # Force CPU to avoid potential GPU/memory issues
        model = FCoref(device='cpu')
        
        # Predict coreference clusters
        preds = model.predict(texts=[text_fragment])
        result = preds[0]
        
        # --- Manual Text Resolution based on successful test ---
        clusters = result.clusters
        char_map = result.char_map
        text_list = list(text_fragment)
        replacements = []
        
        for cluster in clusters:
            # The first mention in the cluster is the antecedent (token indices)
            antecedent_token_span = cluster[0]
            
            # Map token indices to character indices
            # char_map[token_span] returns (text_idx, (start_char, end_char))
            _, (antecedent_start, antecedent_end) = char_map[antecedent_token_span]
            antecedent = text_fragment[antecedent_start:antecedent_end]
            
            # Iterate over all other mentions in the cluster (the anaphors)
            for mention_token_span in cluster[1:]:
                # Ensure the token span is in the char_map (it should be)
                if mention_token_span in char_map:
                    _, (mention_start, mention_end) = char_map[mention_token_span]
                    
                    # We only want to replace pronouns or short mentions, not full names
                    # For simplicity, we replace all non-antecedent mentions for now
                    replacements.append({
                        'start': mention_start,
                        'end': mention_end,
                        'replacement': antecedent
                    })

        # Sort replacements by start index in reverse order
        replacements.sort(key=lambda x: x['start'], reverse=True)

        # Apply replacements
        for rep in replacements:
            start = rep['start']
            end = rep['end']
            replacement = rep['replacement']
            
            # Replace the slice of the text list
            text_list[start:end] = list(replacement)
            
        resolved_text = "".join(text_list)
        # --- End Manual Text Resolution ---

        print(f"Coreference Resolution: {len(clusters)} clusters resolved in the first chapter.")
        return resolved_text

    except Exception as e:
        print(f"Warning: Coreference resolution failed with fastcoref: {e}")
        print("Returning original text fragment.")
        return text_fragment
    finally:
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()

def ner_spacy(text_path, use_coref=False):
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError as e:
        raise SystemExit(
            "SpaCy model 'en_core_web_lg' not found.\n"
            "Install with: python -m spacy download en_core_web_lg"
        ) from e

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = strip_gutenberg_headers(text)

    if use_coref:
        print("Applying Coreference Resolution...")
        # The resolve_coreferences function now handles the text fragmentation
        text_to_process = resolve_coreferences(text)
        print("Coreference Resolution applied.")
    else:
        # If not using coref, still process only the first chapter for a fair comparison
        text_to_process = get_first_chapter(text)


    # Process the text in chunks to avoid memory issues with large models
    # We will process it paragraph by paragraph (split by double newline)
    paragraphs = text_to_process.split('\n\n')
    
    # Initialize global counters and collectors
    characters = set()
    occurrences = defaultdict(list)
    sentence_index = 0
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        # Process the paragraph with the spacy model
        doc = nlp(paragraph)
        
        # Process sentences within the paragraph
        for s in doc.sents:
            # 1. Extract PERSON entities from the sentence
            for ent in s.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    characters.add(name)
                    # Use the global sentence index as the timestamp
                    occurrences[name].append((sentence_index, s.text.strip()))
            
            sentence_index += 1
            
    # Save characters to a file
    # (Removed file saving for brevity, will be handled in main.py or interactions.py)
    
    print(f"Done NER. Found {len(characters)} unique PERSON entities in {sentence_index} sentences.")
    return occurrences

# The ner_stanford and ner_flair functions are removed for brevity and focus on the main task.

if __name__ == "__main__":
    # Example usage (requires a text file)
    # This part is for testing the new function only, not for the final pipeline.
    # The final pipeline will be orchestrated by main.py
    
    # Create a dummy file for testing
    dummy_text = "Elizabeth Bennet was a bright young woman. She lived in Longbourn. Her family was well-known in the neighborhood."
    with open("dummy_test.txt", "w") as f:
        f.write(dummy_text)
        
    # Test coreference resolution
    resolved = resolve_coreferences("dummy_test.txt")
    print("\n--- Coreference Test Result ---")
    print(resolved)
    
    # Clean up
    import os
    os.remove("dummy_test.txt")
