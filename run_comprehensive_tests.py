import json
import os
from improved_processing import get_sentences, resolve_coreferences_improved, detect_interactions_contextual

def main():
    text_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/data/pride_and_prejudice_cleaned.txt"
    char_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/final_characters.json"
    
    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    with open(char_path, "r") as f:
        char_data = json.load(f)
    char_dict = {name: [a.lower() for a in info['aliases']] for name, info in char_data.items()}
    
    output_file = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/test_results_summary.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== PZE Project: Comprehensive Test Results ===\n\n")
        
        # 1. Fragment Size Analysis
        f.write("1. Fragment Size Analysis\n")
        f.write("-" * 30 + "\n")
        for size in [1000, 5000, 10000]:
            fragment = full_text[:size]
            sentences = get_sentences(fragment)
            f.write(f"Fragment Size: {size} chars -> {len(sentences)} sentences.\n")
        f.write("\n")
        
        # 2. Coreference Resolution Example
        f.write("2. Coreference Resolution Example (First 1000 chars)\n")
        f.write("-" * 30 + "\n")
        original_fragment = full_text[:1000]
        resolved_text, cluster_count = resolve_coreferences_improved(original_fragment)
        f.write(f"Clusters found: {cluster_count}\n")
        f.write("Original (excerpt): " + original_fragment[:200].replace('\n', ' ') + "...\n")
        f.write("Resolved (excerpt): " + resolved_text[:200].replace('\n', ' ') + "...\n\n")
        
        # 3. Contextual Interaction Detection (First 20 sentences)
        f.write("3. Contextual Interaction Detection (First 20 sentences)\n")
        f.write("-" * 30 + "\n")
        sentences = get_sentences(full_text[:5000])[:20]
        interactions = detect_interactions_contextual(sentences, char_dict)
        
        for inter in interactions:
            f.write(f"Sentence [{inter['sentence_index']}]: {inter['sentence']}\n")
            f.write(f"Detected Characters: {', '.join(inter['characters'])}\n\n")
            
    print(f"Comprehensive test results saved to {output_file}")

if __name__ == "__main__":
    main()
