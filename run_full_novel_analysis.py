import json
import os
import re
import spacy
from collections import defaultdict

# Load SpaCy model for sentence segmentation
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Increase max_length for large documents
nlp.max_length = 2000000 

def get_sentences_full(text):
    """Segment the entire text into sentences using SpaCy."""
    print("Segmenting full text into sentences (this may take a while)...")
    # Process in chunks to avoid memory issues even with increased max_length
    chunk_size = 100000
    sentences = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        doc = nlp(chunk)
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
    return sentences

def detect_interactions_contextual_full(sentences, character_dict):
    """Run contextual weighting analysis on all sentences."""
    print(f"Analyzing {len(sentences)} sentences for interactions...")
    interactions = []
    context_weights = defaultdict(float)
    
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        current_mentions = set()
        
        for char_name, aliases in character_dict.items():
            for alias in aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', sent_lower):
                    current_mentions.add(char_name)
                    context_weights[char_name] = 1.0
                    break
        
        # Decay
        for char in list(context_weights.keys()):
            if char not in current_mentions:
                context_weights[char] *= 0.7
                if context_weights[char] < 0.1:
                    del context_weights[char]
        
        active_chars = [char for char, weight in context_weights.items() if weight > 0.3]
        
        if len(active_chars) >= 2:
            interactions.append({
                "sentence_index": i,
                "characters": sorted(active_chars)
            })
            
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} sentences...")
            
    return interactions

def main():
    text_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/data/pride_and_prejudice_cleaned.txt"
    char_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/final_characters.json"
    
    if not os.path.exists(text_path):
        print(f"Error: {text_path} not found.")
        return

    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    with open(char_path, "r") as f:
        char_data = json.load(f)
    char_dict = {name: [a.lower() for a in info['aliases']] for name, info in char_data.items()}
    
    # 1. Segment
    sentences = get_sentences_full(full_text)
    
    # 2. Analyze
    interactions = detect_interactions_contextual_full(sentences, char_dict)
    
    # 3. Save Results
    output_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/full_novel_interactions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(interactions, f, indent=2)
    
    # 4. Generate Summary Statistics
    stats_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/full_novel_stats.txt"
    pair_counts = defaultdict(int)
    for inter in interactions:
        chars = inter['characters']
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                pair = tuple(sorted([chars[i], chars[j]]))
                pair_counts[pair] += 1
                
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== Full Novel Interaction Statistics: Pride and Prejudice ===\n\n")
        f.write(f"Total sentences analyzed: {len(sentences)}\n")
        f.write(f"Total interaction events detected: {len(interactions)}\n\n")
        f.write("Top Character Pairs by Interaction Frequency:\n")
        f.write("-" * 45 + "\n")
        for pair, count in sorted_pairs[:20]:
            f.write(f"{pair[0]} <-> {pair[1]}: {count}\n")

    print(f"Analysis complete. Results saved to {output_path} and {stats_path}")

if __name__ == "__main__":
    main()
