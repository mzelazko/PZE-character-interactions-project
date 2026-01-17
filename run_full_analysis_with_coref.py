import json
import os
import re
import spacy
from collections import defaultdict
from fastcoref import FCoref

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Load fastcoref model
print("Loading fastcoref model...")
coref_model = FCoref(device='cpu')

def resolve_coreferences(text_fragment):
    """Resolve coreferences in a text fragment."""
    try:
        preds = coref_model.predict(texts=[text_fragment])
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
            
        return "".join(text_list)
    except Exception as e:
        print(f"Coreference resolution failed for fragment: {e}")
        return text_fragment

def detect_interactions_contextual(sentences, character_dict):
    """Detect interactions using contextual weighting."""
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
                "characters": sorted(active_chars),
                "sentence": sent
            })
            
    return interactions

def main():
    text_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/data/pride_and_prejudice_cleaned.txt"
    char_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/final_characters.json"
    
    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    with open(char_path, "r") as f:
        char_data = json.load(f)
    char_dict = {name: [a.lower() for a in info['aliases']] for name, info in char_data.items()}
    
    # Split by Chapter
    chapters = re.split(r'(CHAPTER [IVXLCDM]+)', full_text)
    
    all_interactions = []
    
    print(f"Found {len(chapters)//2} chapters. Starting analysis...")
    
    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i]
        chapter_content = chapters[i+1] if i+1 < len(chapters) else ""
        full_chapter = chapter_title + chapter_content
        
        print(f"Processing {chapter_title}...")
        
        # 1. Resolve Coreferences
        resolved_chapter = resolve_coreferences(full_chapter)
        
        # 2. Segment into sentences
        doc = nlp(resolved_chapter)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # 3. Detect Interactions
        chapter_interactions = detect_interactions_contextual(sentences, char_dict)
        for inter in chapter_interactions:
            inter['chapter'] = chapter_title
            all_interactions.append(inter)
            
    # Save Results
    output_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/full_analysis_with_coref.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_interactions, f, indent=2)
        
    # Stats
    pair_counts = defaultdict(int)
    for inter in all_interactions:
        chars = inter['characters']
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                pair = tuple(sorted([chars[i], chars[j]]))
                pair_counts[pair] += 1
                
    stats_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/full_analysis_with_coref_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Total interactions detected with fastcoref: {len(all_interactions)}\n\n")
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_pairs[:20]:
            f.write(f"{pair[0]} <-> {pair[1]}: {count}\n")
            
    print(f"Analysis complete. Results saved to {output_path} and {stats_path}")

if __name__ == "__main__":
    main()
