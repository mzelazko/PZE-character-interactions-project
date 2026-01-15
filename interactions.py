import json
import requests
import re
import argparse
from pathlib import Path
from collections import defaultdict, deque
from fastcoref import FCoref

TEXT_PATH = "./data/pride_and_prejudice_cleaned.txt"
CHARACTERS_PATH = "./results/final_characters.json"
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b-it-qat"

def extract_characters():
    """Load character list with variants from final_characters.json"""
    with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
        characters_data = json.load(f)
    
    character_dict = {}
    for char_name, char_info in characters_data.items():
        aliases = [alias.lower() for alias in char_info.get("aliases", [])]
        if aliases:
            character_dict[char_name] = aliases
    
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

# ============================================================================
# METHOD 1: SLIDING WINDOW WITH HEURISTIC PRONOUN RESOLUTION
# ============================================================================

def resolve_pronouns_heuristic(sentence, recent_characters):
    """Simple pronoun resolution using gender and recency"""
    
    # Gendered pronouns
    male_pronouns = {'he', 'him', 'his', 'himself', 'mr'}
    female_pronouns = {'she', 'her', 'hers', 'herself', 'miss', 'mrs'}
    plural_pronouns = {'they', 'them', 'their', 'themselves'}
    
    # Gender mapping for characters
    male_chars = {'Darcy', 'Bingley', 'Bennet', 'Collins', 'Wickham', 'William', 
                  'Charles', 'George', 'Richard', 'John'}
    female_chars = {'Elizabeth', 'Jane', 'Lydia', 'Kitty', 'Catherine', 
                    'Mary', 'Charlotte', 'Lady Catherine', 'Georgiana', 'Caroline',
                    'Maria', 'Lizzy', 'Harriet'}
    
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

def detect_interactions_sliding_window(text, character_dict):
    """Detect interactions with heuristic pronoun resolution"""
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
        pronoun_characters = resolve_pronouns_heuristic(line, recent_characters)
        
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

# ============================================================================
# METHOD 2: FASTCOREF
# ============================================================================

def match_mention_to_character(mention_text, character_dict):
    """Match a coreference mention to a known character"""
    mention_lower = mention_text.lower().strip()
    
    for char_name, aliases in character_dict.items():
        for alias in aliases:
            if alias in mention_lower or mention_lower in alias:
                return char_name
    return None

def detect_interactions_fastcoref(text, character_dict, chunk_size=100):
    """Detect interactions using fastcoref for coreference resolution with chunking"""
    print("Loading fastcoref model...")
    model = FCoref(device='cpu')
    
    # Split text into lines for line number tracking
    lines = text.split('\n')
    
    print(f"Processing {len(lines)} lines in chunks of {chunk_size}...")
    
    # Track character mentions across all chunks
    line_characters = defaultdict(set)
    
    total_chunks = (len(lines) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(0, len(lines), chunk_size):
        chunk_lines = lines[chunk_idx:chunk_idx + chunk_size]
        chunk_text = '\n'.join(chunk_lines)
        
        current_chunk = chunk_idx // chunk_size + 1
        print(f"Processing chunk {current_chunk}/{total_chunks}...")
        
        # Run coreference resolution on this chunk
        try:
            preds = model.predict(texts=[chunk_text])
            
            # Get clusters (each cluster is a list of coreferent mentions)
            clusters = preds[0].get_clusters(as_strings=True)
            
            # Map each cluster to a character
            cluster_to_character = {}
            for cluster_id, mentions in enumerate(clusters):
                # Try to find a character name in the mentions
                for mention in mentions:
                    char = match_mention_to_character(mention, character_dict)
                    if char:
                        cluster_to_character[cluster_id] = char
                        break
            
            # Get token-level clusters for position mapping
            token_clusters = preds[0].get_clusters()
            
            # Build a map from character position to cluster (within chunk)
            char_pos_to_cluster = {}
            for cluster_id, token_positions in enumerate(token_clusters):
                for position_tuple in token_positions:
                    try:
                        if isinstance(position_tuple, (list, tuple)) and len(position_tuple) >= 2:
                            start = int(position_tuple[0])
                            end = int(position_tuple[1])
                            for pos in range(start, end + 1):
                                char_pos_to_cluster[pos] = cluster_id
                        elif isinstance(position_tuple, int):
                            char_pos_to_cluster[position_tuple] = cluster_id
                    except (ValueError, TypeError) as e:
                        # Skip invalid position data
                        continue
            
            # Track character positions in each line of this chunk
            current_pos = 0
            
            for i, line in enumerate(chunk_lines):
                line_num = chunk_idx + i + 1
                line_chars = set()
                
                # Check direct mentions in this line
                direct_mentions = find_character_in_text(line, character_dict)
                line_chars.update(direct_mentions)
                
                # Check for pronouns/mentions resolved by coreference
                line_start = current_pos
                line_end = current_pos + len(line)
                
                for pos in range(line_start, line_end):
                    if pos in char_pos_to_cluster:
                        cluster_id = char_pos_to_cluster[pos]
                        if cluster_id in cluster_to_character:
                            line_chars.add(cluster_to_character[cluster_id])
                
                if len(line_chars) >= 2:
                    line_characters[line_num] = line_chars
                
                current_pos = line_end + 1
        
        except Exception as e:
            print(f"Warning: Error processing chunk {current_chunk}: {e}")
            print("Falling back to direct mention detection for this chunk...")
            for i, line in enumerate(chunk_lines):
                line_num = chunk_idx + i + 1
                direct_mentions = find_character_in_text(line, character_dict)
                if len(direct_mentions) >= 2:
                    line_characters[line_num] = direct_mentions
    
    interactions = []
    for line_num in sorted(line_characters.keys()):
        characters = sorted(line_characters[line_num])
        interaction = tuple([line_num] + characters)
        interactions.append(interaction)
    
    return interactions

# ============================================================================
# METHOD 3: LLM (OLLAMA)
# ============================================================================

def call_ollama(prompt, model=MODEL_NAME):
    """Call local Ollama API with generate endpoint"""
    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            },
            timeout=200
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            print(f"Error: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            return ""
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""

def parse_llm_response(response_text, character_names):
    """Parse LLM response to extract line numbers and character names"""
    interactions = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove common prefixes
        line = re.sub(r'^[-*•]\s*', '', line)
        
        # Try to match "Line X:" or "X:" patterns
        match = re.match(r'(?:Line\s+)?(\d+)\s*[:)\-]\s*(.+)', line, re.IGNORECASE)
        
        if match:
            line_num = int(match.group(1))
            chars_text = match.group(2).strip()
            
            # Extract character names from the text
            found_chars = set()
            chars_text_lower = chars_text.lower()
            
            for char_name in character_names:
                # Check if character name appears in the response
                if char_name.lower() in chars_text_lower or char_name.replace('_', ' ').lower() in chars_text_lower:
                    found_chars.add(char_name)
            
            if len(found_chars) >= 2:
                interactions.append((line_num, found_chars))
    
    return interactions

def detect_interactions_llm(text, character_dict, paragraphs_per_chunk=10):
    """Detect interactions using local LLM, splitting by paragraphs"""
    lines = text.split('\n')
    character_names = list(character_dict.keys())
    
    # Group lines into paragraphs (separated by empty lines)
    paragraphs = []
    current_paragraph = []
    para_start_line = 1
    
    for i, line in enumerate(lines):
        current_line_num = i + 1
        
        if line.strip():  # Non-empty line
            current_paragraph.append((current_line_num, line))
        else:  # Empty line = paragraph boundary
            if current_paragraph:
                paragraphs.append((para_start_line, current_paragraph))
                current_paragraph = []
            para_start_line = current_line_num + 1
    
    if current_paragraph:
        paragraphs.append((para_start_line, current_paragraph))
    
    print(f"Text divided into {len(paragraphs)} paragraphs")
    print(f"Processing in chunks of {paragraphs_per_chunk} paragraphs...\n")
    
    # Prepare character list with aliases for prompt
    char_descriptions = []
    for char_name, aliases in character_dict.items():
        alias_str = ", ".join(aliases[:5])
        char_descriptions.append(f"{char_name} (also: {alias_str})")
    char_list = "; ".join(char_descriptions)
    
    all_interactions = []
    
    # Process paragraphs in chunks
    total_chunks = (len(paragraphs) + paragraphs_per_chunk - 1) // paragraphs_per_chunk
    
    for chunk_idx in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk_paragraphs = paragraphs[chunk_idx:chunk_idx + paragraphs_per_chunk]
        
        # Build chunk text with line numbers and create mapping
        numbered_text = ""
        line_mapping = {}  # Maps virtual line number in chunk to original line number
        virtual_line = 1
        
        for para_start, para_lines in chunk_paragraphs:
            for orig_line_num, line_content in para_lines:
                numbered_text += f"{virtual_line}: {line_content}\n"
                line_mapping[virtual_line] = orig_line_num
                virtual_line += 1
            numbered_text += "\n"
        
        if not numbered_text.strip():
            continue
        
        current_chunk = chunk_idx // paragraphs_per_chunk + 1
        para_range_start = chunk_paragraphs[0][0]
        para_range_end = chunk_paragraphs[-1][0] + len(chunk_paragraphs[-1][1])
        print(f"Processing chunk {current_chunk}/{total_chunks} (paragraphs {chunk_idx + 1}-{min(chunk_idx + paragraphs_per_chunk, len(paragraphs))}, "
              f"lines {para_range_start}-{para_range_end})...")
        
        # Create prompt for LLM
        prompt = f"""Analyze this text excerpt from Pride and Prejudice. Find all lines where TWO OR MORE of these characters interact or are mentioned together: {char_list}

        Text:
        {numbered_text}

        For each line where 2+ characters interact, respond ONLY with the line number and character names in this exact format:
        Line X: Character1, Character2, Character3

        Only list lines with actual interactions (conversations, mentions together, or related actions). Be concise."""

        response = call_ollama(prompt)
        
        if response:
            chunk_interactions = parse_llm_response(response, character_names)
            
            # Convert virtual line numbers to original line numbers
            for virtual_line_num, chars in chunk_interactions:
                if virtual_line_num in line_mapping:
                    orig_line_num = line_mapping[virtual_line_num]
                    all_interactions.append((orig_line_num, sorted(chars)))
            
            print(f"  Found {len(chunk_interactions)} interactions in this chunk")
        else:
            print(f"  Warning: No response from LLM for chunk {current_chunk}")
    
    # Convert to final format and remove duplicates
    interactions_dict = {}
    for line_num, chars in all_interactions:
        if line_num not in interactions_dict:
            interactions_dict[line_num] = set(chars)
        else:
            interactions_dict[line_num].update(chars)
    
    # Convert to tuple format with deduplication of consecutive identical interactions
    interactions = []
    prev_chars = None
    prev_line = None
    
    for line_num in sorted(interactions_dict.keys()):
        characters = sorted(interactions_dict[line_num])
        if len(characters) >= 2:
            # Check if this is the same character set as previous line
            curr_chars = tuple(characters)
            
            if curr_chars != prev_chars or (prev_line is not None and line_num > prev_line + 1):
                interaction = tuple([line_num] + characters)
                interactions.append(interaction)
                prev_chars = curr_chars
                prev_line = line_num
            else:
                prev_line = line_num
    
    return interactions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Detect character interactions in Pride and Prejudice using different methods"
    )
    parser.add_argument(
        "method",
        choices=["1", "2", "3"],
        help="Method: 1 (sliding window), 2 (fastcoref), 3 (LLM/Ollama)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: results/interactions/interactions_[method].txt)"
    )
    parser.add_argument(
        "-t", "--text",
        default=TEXT_PATH,
        help=f"Input text file (default: {TEXT_PATH})"
    )
    
    args = parser.parse_args()
    method = args.method
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Create default output path based on method
        output_dir = Path("./results/interactions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        method_name = {
            "1": "sliding_window",
            "2": "fastcoref",
            "3": "llm"
        }[method]
        
        output_path = str(output_dir / f"interactions_{method_name}.txt")
    
    # Load text and characters
    print("Loading text...")
    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Loading characters...")
    character_dict = extract_characters()
    print(f"Tracking {len(character_dict)} characters\n")
    
    # Run selected method
    if method == '1':
        print("=== METHOD 1: SLIDING WINDOW WITH HEURISTIC PRONOUN RESOLUTION ===\n")
        interactions = detect_interactions_sliding_window(text, character_dict)
    
    elif method == '2':
        print("=== METHOD 2: FASTCOREF COREFERENCE RESOLUTION ===\n")
        interactions = detect_interactions_fastcoref(text, character_dict)
    
    elif method == '3':
        print("=== METHOD 3: LOCAL LLM (OLLAMA) ANALYSIS ===\n")
        print(f"Using model: {MODEL_NAME}")
        print(f"Make sure Ollama is running with: ollama run {MODEL_NAME}\n")
        interactions = detect_interactions_llm(text, character_dict, paragraphs_per_chunk=10)
    
    # Save results
    print(f"\nFound {len(interactions)} interactions")
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for interaction in interactions:
            f.write(", ".join(map(str, interaction)) + "\n")

if __name__ == "__main__":
    main()
