#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy
import requests
from collections import defaultdict, deque
import re
from gender_data import get_static_gender_map

nlp = spacy.load("en_core_web_sm")
def get_gender_map(character_dict):
    gender_map = {}
    static_gender_map = get_static_gender_map()
    
    for name in character_dict.keys():
        lower_name = name.lower()
        
        # Priority 1: Title-based detection (most reliable)
        if any(title in lower_name for title in ["mr.", "sir"]):
            gender_map[name] = "MALE"
        elif any(title in lower_name for title in ["mrs.", "miss", "ms.", "lady"]):
            gender_map[name] = "FEMALE"
        else:
            # Priority 2: Static name lookup
            # Extract the first name (assuming it's the first word without a title)
            first_name = name.split()[0].lower()
            
            if first_name in static_gender_map:
                gender_map[name] = static_gender_map[first_name]
            else:
                # Priority 3: Unknown
                gender_map[name] = "UNKNOWN"
                
    return gender_map

def analyze_interactions_smart(occurrences, character_dict, gender_map):
    print("Analyzing interactions (Method 1: Heuristic Pronoun Resolution)...")
    interactions = defaultdict(list)
    
    # Invert the character_dict for quick lookup: {alias: MasterName}
    alias_to_master = {alias: master for master, aliases in character_dict.items() for alias in aliases}
    
    # Get all sentence indices where characters appear
    sentence_indices = sorted(list(set(idx for occs in occurrences.values() for idx, _ in occs)))
    
    # Create a map of sentence_index -> list of MasterNames in that sentence
    sentence_to_chars = defaultdict(set)
    for master_name, occs in occurrences.items():
        for idx, _ in occs:
            sentence_to_chars[idx].add(master_name)

    # Track the last mentioned characters for pronoun resolution
    last_male = None
    last_female = None
    last_person = None

    for idx in sentence_indices:
        present_chars = sentence_to_chars[idx]
        
        # Update last mentioned characters
        for char_name in present_chars:
            last_person = char_name
            if gender_map.get(char_name) == "MALE":
                last_male = char_name
            elif gender_map.get(char_name) == "FEMALE":
                last_female = char_name

        # Check for pronouns in the sentence text
        # This is a simplified check. A proper implementation would use coreference resolution.
        # We need to find a sentence text from one of the occurrences for the current index
        sentence_text = ""
        for master_name in present_chars:
            for occ_idx, text in occurrences[master_name]:
                if occ_idx == idx:
                    sentence_text = text
                    break
            if sentence_text:
                break
        
        lower_sent = sentence_text.lower()
        
        if "he" in lower_sent.split() and last_male:
            present_chars.add(last_male)
        if "she" in lower_sent.split() and last_female:
            present_chars.add(last_female)

        if len(present_chars) > 1:
            # Sort for consistent ordering
            char_tuple = tuple(sorted(list(present_chars)))
            interactions[char_tuple].append(idx)

    # Save to file
    with open("interactions_smart.txt", "w", encoding="utf-8") as f:
        for chars, indices in sorted(interactions.items(), key=lambda item: len(item[1]), reverse=True):
            for idx in indices:
                f.write(f"({idx}, {', '.join(chars)})\n")
    print("Method 1 results saved to interactions_smart.txt")
    return interactions

def analyze_interactions_window(occurrences, character_dict):
    print("Analyzing interactions (Method 2: Sliding Window)...")
    interactions = defaultdict(list)
    
    # Flatten all occurrences into a single list of (sentence_idx, MasterName)
    all_char_occurrences = []
    for master_name, occs in occurrences.items():
        for idx, _ in occs:
            all_char_occurrences.append((idx, master_name))
    
    # Sort by sentence index
    all_char_occurrences.sort()
    
    # Use a sliding window to find co-occurrences
    window_size = 5 # Number of sentences to look forward/backward
    
    for i, (idx1, char1) in enumerate(all_char_occurrences):
        # Look ahead in the window
        for j in range(i + 1, len(all_char_occurrences)):
            idx2, char2 = all_char_occurrences[j]
            
            # If the next character is outside the window, break
            if idx2 - idx1 > window_size:
                break
            
            # If it's a different character, record an interaction
            if char1 != char2:
                # Use the sentence index of the first character as the timestamp
                interaction_tuple = tuple(sorted((char1, char2)))
                interactions[interaction_tuple].append(idx1)

    # Save to file
    with open("interactions_window.txt", "w", encoding="utf-8") as f:
        for chars, indices in sorted(interactions.items(), key=lambda item: item[1], reverse=True):
            # We only write unique indices for each pair
            for idx in sorted(list(set(indices))):
                f.write(f"({idx}, {', '.join(chars)})\n")
    print("Method 2 results saved to interactions_window.txt")
    return interactions

def generate_statistics(interactions_dict, filename="interaction_statistics.txt"):
    pair_counts = defaultdict(int)
    for chars, indices in interactions_dict.items():
        # Consider only pairs for simplicity
        if len(chars) == 2:
            pair_counts[chars] += len(indices)
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n==== Interaction Pair Counts ====\n")
        for pair, count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
            f.write(f"{pair[0]} - {pair[1]}: {count} interactions\n")

# The main function now accepts the occurrences dictionary
def main(occurrences):
    # --- Re-implementing Alias Resolution and Master Name Mapping ---
    
    # 1. Get all raw mentions and their frequencies
    all_mentions = defaultdict(int)
    for name, occs in occurrences.items():
        all_mentions[name] = len(occs)
        
    # 2. Sort all mentions by frequency
    sorted_names = sorted(all_mentions.items(), key=lambda item: item[1], reverse=True)
    
    # 3. Use the top 15 names as the basis for the main characters
    main_characters = [name for name, count in sorted_names[:15]]
    
    final_character_dict = defaultdict(list)
    grouped_names = set()
    alias_to_master = {}
    
    # Iterate through the main characters (most frequent first)
    for master_name in main_characters:
        if master_name in grouped_names:
            continue
            
        # Prefer multi-word names as master names
        if len(master_name.split()) < 2:
            # Skip single-word names for now, they will be handled later if they are not aliases
            continue
            
        # Normalize the master name (remove titles)
        normalized_master = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', master_name, flags=re.I).lower()
        
        variants = []
        
        # Iterate through all frequent names to find aliases
        for alias_name, count in sorted_names:
            if alias_name in grouped_names:
                continue
            
            normalized_alias = re.sub(r'^(mr|mrs|miss|ms|lady|sir)\.?\s+', '', alias_name, flags=re.I).lower()
            
            # Heuristic 1: Full name contains the alias name (e.g., 'Miss Elizabeth Bennet' contains 'Elizabeth')
            if normalized_alias in normalized_master and len(normalized_alias) > 2:
                variants.append(alias_name.lower())
                grouped_names.add(alias_name)
                
            # Heuristic 2: Alias name is a single word and is the surname of the master name (e.g., 'Bennet' for 'Mr. Bennet')
            if len(alias_name.split()) == 1 and alias_name.lower() in normalized_master.split():
                variants.append(alias_name.lower())
                grouped_names.add(alias_name)

        # Ensure the master name itself is included
        variants.append(master_name.lower())
        
        if variants:
            final_character_dict[master_name] = list(set(variants))
            for alias in variants:
                alias_to_master[alias] = master_name
            
    # Now, handle the remaining single-word names that are frequent but were not grouped (e.g., 'Darcy', 'Elizabeth')
    for name, count in sorted_names[:15]:
        if name not in grouped_names and len(name.split()) == 1 and name.lower() not in ('mr', 'mrs', 'miss', 'sir', 'lady', 'ms'):
            final_character_dict[name] = [name.lower()]
            alias_to_master[name.lower()] = name
            
    # Final step: Re-normalize the master names to remove titles for cleaner keys
    new_dict = {}
    for master_name, variants in final_character_dict.items():
        clean_name = re.sub(r'^(mr|mrs|miss|lady|sir)\.?\s+', '', master_name, flags=re.I).strip()
        if not clean_name:
            clean_name = master_name
            
        # Use the most frequent single name in the variants as the clean name if available
        single_names = [v for v in variants if len(v.split()) == 1 and v not in ('mr', 'mrs', 'miss', 'sir', 'lady', 'ms')]
        
        if single_names:
            # Find the most frequent one among the single names
            # This is a simplification, we'll just use the first one for now
            new_dict[single_names[0].capitalize()] = variants
        else:
            new_dict[master_name] = variants
            
    # Rebuild alias_to_master with the new master names
    alias_to_master = {}
    for master, aliases in new_dict.items():
        for alias in aliases:
            alias_to_master[alias] = master
            
    # Create the new occurrences dictionary with Master Names as keys
    master_occurrences = defaultdict(list)
    
    for raw_name, occs in occurrences.items():
        # Find the master name for the raw mention
        master_name = alias_to_master.get(raw_name.lower())
        
        if master_name:
            # Use the capitalized master name from new_dict as the key
            master_occurrences[master_name].extend(occs)
            
    occurrences = master_occurrences
    character_dict = new_dict
    
    # --- End Re-implementing Alias Resolution and Master Name Mapping ---
    
    print(f"Found {len(character_dict)} main characters after alias resolution.")
    
    # Get gender map for pronoun resolution
    gender_map = get_gender_map(character_dict)
    
    # Clear previous statistics file
    with open("interaction_statistics.txt", "w", encoding="utf-8") as f:
        f.write("Interaction Analysis Report\n")

    # --- Method 1: Smart (Heuristic) Analysis ---
    interactions1 = analyze_interactions_smart(occurrences, character_dict, gender_map)
    generate_statistics(interactions1)

    # --- Method 2: Sliding Window Analysis ---
    interactions2 = analyze_interactions_window(occurrences, character_dict)
    with open("interaction_statistics.txt", "a", encoding="utf-8") as f:
        f.write("\n\n==== Sliding Window Method ====")
    generate_statistics(interactions2)

    print("\nAnalysis complete. Results are in interaction_statistics.txt, interactions_smart.txt, and interactions_window.txt")

if __name__ == "__main__":
    
    pass
