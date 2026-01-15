import spacy
import json
import re
from collections import defaultdict, deque
from fastcoref import FCoref
import os

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

def get_sentences(text):
    """Improve sentence segmentation using SpaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def process_fragments(text, fragment_sizes=[1000, 5000, 10000]):
    """Process text with different fragment sizes."""
    results = {}
    for size in fragment_sizes:
        fragment = text[:size]
        sentences = get_sentences(fragment)
        results[size] = {
            "char_count": len(fragment),
            "sentence_count": len(sentences),
            "sentences": sentences
        }
    return results

def resolve_coreferences_improved(text_fragment):
    """Improved coreference resolution with manual text replacement."""
    try:
        model = FCoref(device='cpu')
        preds = model.predict(texts=[text_fragment])
        result = preds[0]
        
        clusters = result.clusters
        char_map = result.char_map
        text_list = list(text_fragment)
        replacements = []
        
        for cluster in clusters:
            antecedent_token_span = cluster[0]
            _, (antecedent_start, antecedent_end) = char_map[antecedent_token_span]
            antecedent = text_fragment[antecedent_start:antecedent_end]
            
            for mention_token_span in cluster[1:]:
                if mention_token_span in char_map:
                    _, (mention_start, mention_end) = char_map[mention_token_span]
                    replacements.append({
                        'start': mention_start,
                        'end': mention_end,
                        'replacement': antecedent
                    })

        replacements.sort(key=lambda x: x['start'], reverse=True)
        for rep in replacements:
            text_list[rep['start']:rep['end']] = list(rep['replacement'])
            
        return "".join(text_list), len(clusters)
    except Exception as e:
        print(f"Coreference resolution failed: {e}")
        return text_fragment, 0

def detect_interactions_contextual(sentences, character_dict, window_size=3):
    """New method: Contextual Weighting for interactions."""
    interactions = []
    # Track character mentions with weights that decay over time
    context_weights = defaultdict(float)
    
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        current_mentions = set()
        
        for char_name, aliases in character_dict.items():
            for alias in aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', sent_lower):
                    current_mentions.add(char_name)
                    context_weights[char_name] = 1.0 # Reset weight on direct mention
                    break
        
        # Decay existing weights
        for char in list(context_weights.keys()):
            if char not in current_mentions:
                context_weights[char] *= 0.7 # Decay factor
                if context_weights[char] < 0.1:
                    del context_weights[char]
        
        # Identify active characters in context
        active_chars = [char for char, weight in context_weights.items() if weight > 0.3]
        
        if len(active_chars) >= 2:
            interactions.append({
                "sentence_index": i,
                "sentence": sent,
                "characters": sorted(active_chars)
            })
            
    return interactions

def run_tests():
    """Run short tests to demonstrate output."""
    test_sentences = [
        "Elizabeth Bennet was walking in the garden.",
        "She met Mr. Darcy near the bridge.",
        "He looked at her with a mixture of pride and admiration.",
        "Jane and Bingley were already at the house."
    ]
    
    char_dict = {
        "Elizabeth": ["elizabeth", "elizabeth bennet", "she", "her"],
        "Darcy": ["darcy", "mr. darcy", "he", "him"],
        "Jane": ["jane"],
        "Bingley": ["bingley"]
    }
    
    print("\n--- Test Case: Contextual Interaction Detection ---")
    interactions = detect_interactions_contextual(test_sentences, char_dict)
    for inter in interactions:
        print(f"Sent [{inter['sentence_index']}]: {inter['sentence']}")
        print(f"Characters: {', '.join(inter['characters'])}\n")

if __name__ == "__main__":
    # Load cleaned text
    text_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/data/pride_and_prejudice_cleaned.txt"
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        # 1. Test different fragment sizes
        print("Testing different fragment sizes...")
        fragment_results = process_fragments(full_text)
        for size, data in fragment_results.items():
            print(f"Size {size}: {data['sentence_count']} sentences found.")
            
        # 2. Run coreference on a small fragment
        print("\nRunning coreference resolution on first 2000 chars...")
        resolved_text, cluster_count = resolve_coreferences_improved(full_text[:2000])
        print(f"Resolved {cluster_count} clusters.")
        
        # 3. Run contextual interaction detection
        print("\nRunning contextual interaction detection on first 50 sentences...")
        sentences = get_sentences(full_text[:10000])[:50]
        
        # Load character dict from json
        with open("/home/ubuntu/pze_project/PZE-character-interactions-project-main/final_characters.json", "r") as f:
            char_data = json.load(f)
        char_dict = {name: [a.lower() for a in info['aliases']] for name, info in char_data.items()}
        
        interactions = detect_interactions_contextual(sentences, char_dict)
        print(f"Found {len(interactions)} contextual interactions.")
        
        # 4. Run short tests
        run_tests()
    else:
        print(f"File not found: {text_path}")
        run_tests()
