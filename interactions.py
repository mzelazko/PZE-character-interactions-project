import spacy
import requests
from collections import defaultdict, deque
import re

nlp = spacy.load("en_core_web_sm")

def download_book(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Attempt to strip Gutenberg headers/footers
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***")
    if start != -1:
        # Try to find the actual start of the book content after the header
        start_content = text.find("CHAPTER 1", start)
        if start_content == -1:
            start_content = text.find("CHAPTER I", start)
        if start_content != -1:
            start = start_content
    
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***")
    if end == -1: # Fallback for different formatting
        end = text.find("End of the Project Gutenberg")
        
    return text[start:end] if start != -1 and end != -1 else text

def extract_characters(text_path):
    """Extracts main characters and their aliases from the novel text."""
    
    all_mentions = defaultdict(int)
    
    try:
        with open("./results/spacy_occurrences.txt", "r", encoding="utf-8") as f:
            current_name = None
            for line in f:
                if line.startswith("===="):
                    current_name = line.strip().strip("=").strip()
                elif current_name:
                    all_mentions[current_name] += 1
    except FileNotFoundError:
        print("Error: spacy_occurrences.txt not found. Run character_extraction.ner_spacy first.")
        return {}

    top_mentions = sorted(all_mentions.items(), key=lambda item: item[1], reverse=True)[:15]
    
    character_dict = {}
    
    master_names = [name for name, count in top_mentions if len(name.split()) > 1] # Prefer full names as master names
    if not master_names:
        master_names = [name for name, count in top_mentions]

    
    potential_master_names = [name for name, count in top_mentions]

    final_character_dict = defaultdict(list)
    
    # Start with the most frequent names
    sorted_names = sorted(all_mentions.items(), key=lambda item: item[1], reverse=True)
    
    # Use the top 15 names as the basis for the main characters
    main_characters = [name for name, count in sorted_names[:15]]
    
    grouped_names = set()
    
    # Iterate through the main characters (most frequent first)
    for master_name in main_characters:
        if master_name in grouped_names:
            continue
            
        # Normalize the master name for comparison (e.g., remove titles and punctuation)
        normalized_master = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', master_name, flags=re.I).lower()
        
        # This will be the list of variants for the current master name
        variants = []
        
        # Iterate through all frequent names to find aliases
        for alias_name, count in sorted_names:
            # Skip if already grouped
            if alias_name in grouped_names:
                continue
            
            # Normalize the alias name for comparison
            normalized_alias = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', alias_name, flags=re.I).lower()
            
            # Alias resolution heuristic:
            # If the normalized alias is a substring of the normalized master name,
            # OR the normalized master name is a substring of the normalized alias,
            # they are likely the same person.
            # We also require a minimum length to avoid grouping single letters or very short strings.
            if (normalized_alias in normalized_master and len(normalized_alias) > 2) or \
               (normalized_master in normalized_alias and len(normalized_master) > 2):
                
                # Exclude common titles as standalone names
                if len(alias_name.split()) == 1 and alias_name.lower() in ('mr', 'mrs', 'miss', 'sir', 'lady'):
                    continue
                
                # Add the lowercased alias name to the variants list
                variants.append(alias_name.lower())
                grouped_names.add(alias_name)
        
        # Ensure the master name itself is included in its own variants list
        variants.append(master_name.lower())
        
        if variants:
            # The key is the master name (the most frequent one we started with)
            final_character_dict[master_name] = list(set(variants))
            
    # Final cleanup: remove single-word names that are just titles or too generic
    # This is a temporary fix for P&P. A general solution would be more complex.
    # For now, let's keep only names that are multi-word or are single names that are not common titles.
    
    # Let's re-evaluate the grouping. The issue is that 'Bennet' is a common surname.
    # A proper solution is to use a more sophisticated clustering algorithm.
    # For a simple, reusable fix, we will only group names if one is a full name and the other is a single name.
    
    # Let's simplify the alias resolution to only group names that are a full name and a single name.
    
    # Re-initialize the grouping with a more conservative approach.
    final_character_dict = defaultdict(list)
    grouped_names = set()
    
    # 1. Select the top 15 most frequent names as potential master names
    main_characters = [name for name, count in sorted_names[:15]]
    
    for master_name in main_characters:
        if master_name in grouped_names:
            continue
            
        # Prefer multi-word names as master names
        if len(master_name.split()) < 2:
            continue
            
        # Normalize the master name (remove titles)
        normalized_master = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', master_name, flags=re.I).lower()
        
        variants = []
        
        for alias_name, count in sorted_names:
            if alias_name in grouped_names:
                continue
            
            normalized_alias = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', alias_name, flags=re.I).lower()
            
            # Heuristic 1: Full name contains the alias name (e.g., 'Miss Elizabeth Bennet' contains 'Elizabeth')
            if normalized_alias in normalized_master and len(normalized_alias) > 2:
                variants.append(alias_name.lower())
                grouped_names.add(alias_name)
                
            # Heuristic 2: Alias name is a single word and is the surname of the master name (e.g., 'Bennet' for 'Mr. Bennet')
            # This is risky but necessary for P&P.
            if len(alias_name.split()) == 1 and alias_name.lower() in normalized_master.split():
                variants.append(alias_name.lower())
                grouped_names.add(alias_name)

        # Ensure the master name itself is included
        variants.append(master_name.lower())
        
        if variants:
            # Use the most frequent variant as the master key (which is master_name here)
            final_character_dict[master_name] = list(set(variants))
            
    # Now, handle the remaining single-word names that are frequent but were not grouped (e.g., 'Darcy', 'Elizabeth')
    for name, count in sorted_names[:15]:
        if name not in grouped_names and len(name.split()) == 1 and name.lower() not in ('mr', 'mrs', 'miss', 'sir', 'lady'):
            final_character_dict[name] = [name.lower()]
            
    # Re-normalize the master names to remove titles for cleaner keys,
    # but keep the original variant list.
    new_dict = {}
    for master_name, variants in final_character_dict.items():
        clean_name = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', master_name, flags=re.I).strip()
        if not clean_name:
            clean_name = master_name # Fallback if only a title was present
            
        # Use the most frequent single name in the variants as the clean name if available
        single_names = [v for v in variants if len(v.split()) == 1]
        if single_names:
            # Find the single name with the highest frequency (closest to the top of sorted_names)
            best_single_name = sorted(single_names, key=lambda x: [i for i, (n, c) in enumerate(sorted_names) if n.lower() == x][0])[0]
            clean_name = best_single_name.capitalize()
            
        new_dict[clean_name] = variants
        
    return new_dict

    # Re-normalize the master names to remove titles for cleaner keys,
    # but keep the original variant list.
    new_dict = {}
    for master_name, variants in final_character_dict.items():
        clean_name = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', master_name, flags=re.I).strip()
        if not clean_name:
            clean_name = master_name # Fallback if only a title was present
        new_dict[clean_name] = variants
        
    return new_dict


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

def resolve_pronouns_heuristic(sentence, recent_characters, character_dict, gender_map):
    """Simple pronoun resolution using gender and recency"""
    
    # Gendered pronouns
    male_pronouns = {'he', 'him', 'his', 'himself', 'mr'}
    female_pronouns = {'she', 'her', 'hers', 'herself', 'miss', 'mrs'}
    plural_pronouns = {'they', 'them', 'their', 'themselves'}
    
    # Gender mapping for characters (now dynamically generated or empty)
    male_chars = {k for k, v in gender_map.items() if v == 'MALE'}
    female_chars = {k for k, v in gender_map.items() if v == 'FEMALE'} 

    
    words = set(re.findall(r'\b\w+\b', sentence.lower()))
    resolved = set()
    
    # Check for gendered pronouns
    has_male_pronoun = bool(words & male_pronouns)
    has_female_pronoun = bool(words & female_pronouns)
    has_plural_pronoun = bool(words & plural_pronouns)
    
    # Resolve based on recent context (most recent character of the correct gender)
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
    
    # Simple plural resolution: assume the two most recent characters are the 'they'
    if has_plural_pronoun and len(recent_characters) >= 2:
        # Only add if they are not already in resolved set
        for char in list(recent_characters)[:2]:
            resolved.add(char)
    
    return resolved

def detect_interactions_smart(text, character_dict, gender_map):
    """Detect interactions with pronoun resolution"""
    interactions = []
    # Split by line, but treat empty lines as context breaks
    lines = text.split('\n')
    
    # Keep track of recently mentioned characters (sliding window)
    # deque stores the character keys (e.g., 'Elizabeth')
    recent_characters = deque(maxlen=5)
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            # Clear context on paragraph break
            recent_characters.clear()
            continue
        
        # Find directly mentioned characters
        direct_mentions = find_character_in_text(line, character_dict)
        
        # Try to resolve pronouns
        pronoun_characters = resolve_pronouns_heuristic(line, recent_characters, character_dict, gender_map)
        
        # Combine direct and resolved
        all_characters = direct_mentions | pronoun_characters
        
        newly_mentioned = []
        for char in direct_mentions:
            if char in recent_characters:
                recent_characters.remove(char)
            newly_mentioned.append(char)
			for char in reversed(newly_mentioned):
            recent_characters.appendleft(char)

        # Record interaction if 2+ characters are present in the line/context
        if len(all_characters) >= 2:
            interaction = tuple([line_num] + sorted(list(all_characters)))
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
            interaction = tuple([i + 1] + sorted(list(characters)))
            if not interactions or interactions[-1] != interaction:
                interactions.append(interaction)
    
    return interactions

def analyze_interactions(interactions, character_dict):
    """Print interaction statistics and save to a file"""
    char_pairs = defaultdict(int)
    
    for interaction in interactions:
        line_num = interaction[0]
        characters = interaction[1:]
        
        # Count pairwise interactions
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                pair = tuple(sorted([char1, char2]))
                char_pairs[pair] += 1
    
    # Save statistics to a file
    with open("interaction_statistics.txt", "a", encoding="utf-8") as f:
        f.write("\n==================================================\n")
        f.write(f"Interaction Analysis for {len(interactions)} interactions\n")
        f.write("==================================================\n")
        
        f.write("\nTop character pairs (by interaction count):\n")
        for pair, count in sorted(char_pairs.items(), key=lambda x: x[1], reverse=True)[:15]:
            f.write(f"  {pair[0]} <-> {pair[1]}: {count} interactions\n")
            
    print("\nTop character pairs (by interaction count):")
    for pair, count in sorted(char_pairs.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {pair[0]} <-> {pair[1]}: {count} interactions")

def get_gender_map(character_dict):
    """Generates a gender map based on titles and a simple heuristic for single names."""
    gender_map = {}
    
    # Simple list of common names and their gender for P&P characters as a fallback
    # In a general solution, this would be a large lookup table or an external service.
    name_to_gender = {
        'elizabeth': 'FEMALE', 'darcy': 'MALE', 'jane': 'FEMALE', 'bingley': 'MALE', 
        'bennet': 'UNKNOWN', 'collins': 'MALE', 'wickham': 'MALE', 'lydia': 'FEMALE', 
        'catherine': 'FEMALE', 'mary': 'FEMALE', 'charlotte': 'FEMALE', 'georgiana': 'FEMALE', 
        'caroline': 'FEMALE', 'fitzwilliam': 'MALE', 'kitty': 'FEMALE', 'lizzy': 'FEMALE',
        'gardiner': 'UNKNOWN', 'george': 'MALE', 'allen': 'UNKNOWN'
    }
    
    for master_name in character_dict.keys():
        name_lower = master_name.lower()
        
        # 1. Title-based detection (most reliable)
        if 'mr.' in name_lower or 'mr ' in name_lower or 'sir' in name_lower or 'lord' in name_lower:
            gender_map[master_name] = 'MALE'
        elif 'mrs.' in name_lower or 'mrs ' in name_lower or 'miss' in name_lower or 'lady' in name_lower:
            gender_map[master_name] = 'FEMALE'
        
        # 2. Single-name fallback (heuristic)
        else:
            # Extract the first name if it's a multi-word name
            first_name = name_lower.split()[0] if len(name_lower.split()) > 1 else name_lower
            
            # Check against the lookup table
            if first_name in name_to_gender:
                gender_map[master_name] = name_to_gender[first_name]
            else:
                gender_map[master_name] = 'UNKNOWN'
        
    return gender_map

def main():
    text_path = "./data/pride_and_prejudice.txt"
    print(f"Reading {text_path}...")
    text = download_book(text_path)
    
    character_dict = extract_characters(text_path)
    print(f"Tracking {len(character_dict)} main characters")
    
    # Clear previous statistics file
    with open("interaction_statistics.txt", "w", encoding="utf-8") as f:
        f.write("Interaction Analysis Report\n")
    
    gender_map = get_gender_map(character_dict)
    print(f"Generated Gender Map: {gender_map}")
    
    print("\n=== Method 1: Heuristic pronoun resolution ===")
    interactions_smart = detect_interactions_smart(text, character_dict, gender_map)
    
    with open("interactions_smart.txt", "w", encoding="utf-8") as f:
        for interaction in interactions_smart:
            f.write("(" + ", ".join(map(str, interaction)) + ")\n")
    
    print(f"✓ Found {len(interactions_smart)} interactions")
    print("\nSample interactions (first 20):")
    for interaction in interactions_smart[:20]:
        print(interaction)
    
    analyze_interactions(interactions_smart, character_dict)
    
    print("\n\n=== Method 2: Sliding window (simple) ===")
    interactions_window = detect_interactions_window(text, character_dict, window_size=8)
    
    with open("interactions_window.txt", "w", encoding="utf-8") as f:
        for interaction in interactions_window:
            f.write("(" + ", ".join(map(str, interaction)) + ")\n")
    
    print(f"✓ Found {len(interactions_window)} interactions")
    analyze_interactions(interactions_window, character_dict)

if __name__ == "__main__":
    main()
