import spacy
import requests
from collections import defaultdict, deque
import re

nlp = spacy.load("en_core_web_sm")

def extract_characters():
    """Manual character list with variants for Pride and Prejudice"""
    character_dict = {
        "Elizabeth": ["elizabeth", "lizzy", "eliza", "miss elizabeth"],
        "Darcy": ["darcy", "mr. darcy", "mr darcy", "fitzwilliam"],
        "Jane": ["jane bennet", "jane", "miss bennet"],
        "Bingley": ["bingley", "mr. bingley", "mr bingley", "charles bingley"],
        "Mr_Bennet": ["mr. bennet", "mr bennet"],
        "Mrs_Bennet": ["mrs. bennet", "mrs bennet"],
        "Collins": ["collins", "mr. collins", "mr collins"],
        "Wickham": ["wickham", "mr. wickham"],
        "Lydia": ["lydia"],
        "Catherine": ["catherine", "kitty"],
        "Mary": ["mary bennet"],
        "Charlotte": ["charlotte", "charlotte lucas"],
        "Lady_Catherine": ["lady catherine"],
        "Georgiana": ["georgiana"],
        "Caroline": ["caroline", "miss bingley"]
    }
    
    return character_dict

def find_character_in_text(text, character_dict):
    """Find which character(s) are mentioned in text"""
    text_lower = text.lower()
    found = set()
    
    for char_name, variants in character_dict.items():
        for variant in variants:
            if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                found.add(char_name)
                break
    
    return found

def resolve_pronouns_heuristic(sentence, recent_characters, character_dict):
    """Simple pronoun resolution using gender and recency"""
    
    # Gendered pronouns
    male_pronouns = {'he', 'him', 'his', 'himself', 'mr'}
    female_pronouns = {'she', 'her', 'hers', 'herself', 'miss', 'mrs'}
    plural_pronouns = {'they', 'them', 'their', 'themselves'}
    
    # Gender mapping for characters
    male_chars = {'Darcy', 'Bingley', 'Mr_Bennet', 'Collins', 'Wickham'}
    female_chars = {'Elizabeth', 'Jane', 'Mrs_Bennet', 'Lydia', 'Catherine', 
                    'Mary', 'Charlotte', 'Lady_Catherine', 'Georgiana', 'Caroline'}
    
    words = set(re.findall(r'\b\w+\b', sentence.lower()))
    resolved = set()
    
    # Check for gendered pronouns
    has_male_pronoun = bool(words & male_pronouns)
    has_female_pronoun = bool(words & female_pronouns)
    has_plural_pronoun = bool(words & plural_pronouns)
    
    # Resolve based on recent context
    if has_male_pronoun:
        for char in recent_characters:
            if char in male_chars:
                resolved.add(char)
                break
    
    if has_female_pronoun:
        for char in recent_characters:
            if char in female_chars:
                resolved.add(char)
                break
    
    if has_plural_pronoun and len(recent_characters) >= 2:
        resolved.update(list(recent_characters)[:2])
    
    return resolved

def detect_interactions_smart(text, character_dict):
    """Detect interactions with pronoun resolution"""
    interactions = []
    lines = text.split('\n')
    
    # Keep track of recently mentioned characters (sliding window)
    recent_characters = deque(maxlen=5)
    
    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        
        # Find directly mentioned characters
        direct_mentions = find_character_in_text(line, character_dict)
        
        # Try to resolve pronouns
        pronoun_characters = resolve_pronouns_heuristic(line, recent_characters, character_dict)
        
        # Combine direct and resolved
        all_characters = direct_mentions | pronoun_characters
        
        # Update recent characters
        for char in direct_mentions:
            if char in recent_characters:
                recent_characters.remove(char)
            recent_characters.appendleft(char)
        
        # Record interaction if 2+ characters
        if len(all_characters) >= 2:
            interaction = tuple([line_num] + sorted(all_characters))
            interactions.append(interaction)
    
    return interactions

def detect_interactions_window(text, character_dict, window_size=5):
    """Simpler approach: sliding window of lines"""
    interactions = []
    lines = text.split('\n')
    
    for i in range(len(lines)):
        # Get window of lines
        window_start = max(0, i - window_size)
        window_end = min(len(lines), i + window_size + 1)
        window_text = ' '.join(lines[window_start:window_end])
        
        # Find characters in window
        characters = find_character_in_text(window_text, character_dict)
        
        # Record if 2+ characters present
        if len(characters) >= 2:
            interaction = tuple([i + 1] + sorted(characters))
            if not interactions or interactions[-1] != interaction:
                interactions.append(interaction)
    
    return interactions

def analyze_interactions(interactions, character_dict):
    """Print interaction statistics"""
    char_pairs = defaultdict(int)
    
    for interaction in interactions:
        line_num = interaction[0]
        characters = interaction[1:]
        
        # Count pairwise interactions
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                pair = tuple(sorted([char1, char2]))
                char_pairs[pair] += 1
    
    print("\nTop character pairs (by interaction count):")
    for pair, count in sorted(char_pairs.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {pair[0]} <-> {pair[1]}: {count} interactions")

def main():
    text_path = "./data/pride_and_prejudice_cleaned.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    character_dict = extract_characters()
    print(f"Tracking {len(character_dict)} main characters")
    
    print("\n=== Method 1: Heuristic pronoun resolution ===")
    interactions_smart = detect_interactions_smart(text, character_dict)
    
    with open("interactions_smart.txt", "w", encoding="utf-8") as f:
        for interaction in interactions_smart:
            f.write(", ".join(map(str, interaction)) + "\n")
    
    print(f"✓ Found {len(interactions_smart)} interactions")
    print("\nSample interactions:")
    for interaction in interactions_smart[:20]:
        print(interaction)
    
    analyze_interactions(interactions_smart, character_dict)
    
    print("\n\n=== Method 2: Sliding window (simple) ===")
    interactions_window = detect_interactions_window(text, character_dict, window_size=8)
    
    with open("interactions_window.txt", "w", encoding="utf-8") as f:
        for interaction in interactions_window:
            f.write(", ".join(map(str, interaction)) + "\n")
    
    print(f"✓ Found {len(interactions_window)} interactions")
    analyze_interactions(interactions_window, character_dict)

if __name__ == "__main__":
    main()