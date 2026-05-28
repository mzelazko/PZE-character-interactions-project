import json
import requests
import re
import argparse
import os
from pathlib import Path
from collections import defaultdict, deque

# Optional dependencies with fallback
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from fastcoref import FCoref
    HAS_FASTCOREF = True
except ImportError:
    HAS_FASTCOREF = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

TEXT_PATH = "./data/pride_and_prejudice.txt"
CHARACTERS_PATH = "./results/final_characters.json"
CORRECT_CHARACTERS_PATH = "./results/character_extraction/correct_character_list.txt"
BLACKLIST_PATH = "./data/universal_blacklist.json"
GENDER_MAP_PATH = "./data/universal_gender_map.json"
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b-it-qat" #llama3.1:8b, gemma3:12b-it-qat, qwen3:14b

def load_universal_resources():
    """Load blacklist and gender maps from external files for universality"""
    blacklist = set()
    if os.path.exists(BLACKLIST_PATH):
        with open(BLACKLIST_PATH, "r") as f:
            data = json.load(f)
            for category in data.values():
                blacklist.update([item.lower() for item in category])
    
    gender_map = {"male": set(), "female": set()}
    if os.path.exists(GENDER_MAP_PATH):
        with open(GENDER_MAP_PATH, "r") as f:
            data = json.load(f)
            gender_map["male"].update(data.get("male_names", []))
            gender_map["male"].update(data.get("male_titles", []))
            gender_map["female"].update(data.get("female_names", []))
            gender_map["female"].update(data.get("female_titles", []))
            
    return blacklist, gender_map

def extract_characters(blacklist=None):
    """Load character list with variants from final_characters.json"""
    if blacklist is None: blacklist = set()
    with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
        characters_data = json.load(f)
    
    character_dict = {}
    for char_name, char_info in characters_data.items():
        if char_name.lower() in blacklist:
            continue
        aliases = [alias.lower() for alias in char_info.get("aliases", []) 
                   if alias.lower() not in blacklist and len(alias) > 2]
        if aliases:
            character_dict[char_name] = aliases
    
    return character_dict

def extract_characters_from_txt(blacklist=None):
    """Load character list from correct_character_list.txt"""
    if blacklist is None: blacklist = set()
    with open(CORRECT_CHARACTERS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    characters = [line.strip() for line in lines if line.strip() and line.strip().lower() not in blacklist]
    
    return characters


def find_character_in_text(text, character_dict):
    """Find which character(s) are mentioned in text"""
    text_lower = text.lower()
    found = set()
    
    for char_name, variants in character_dict.items():
        # Check canonical name too
        if re.search(r'\b' + re.escape(char_name.lower()) + r'\b', text_lower):
            found.add(char_name)
            continue
        for variant in variants:
            if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                found.add(char_name)
                break
    
    return found

# ============================================================================
# METHOD 1: SLIDING WINDOW WITH HEURISTIC PRONOUN RESOLUTION
# ============================================================================

def resolve_pronouns_heuristic(sentence, recent_characters, gender_map):
    """Simple pronoun resolution using gender and recency"""
    
    # Gendered pronouns
    male_pronouns = {'he', 'him', 'his', 'himself'}
    female_pronouns = {'she', 'her', 'hers', 'herself'}
    plural_pronouns = {'they', 'them', 'their', 'themselves'}
    
    words = set(re.findall(r'\b\w+\b', sentence.lower()))
    resolved = set()
    
    # Check for gendered pronouns
    has_male_pronoun = bool(words & male_pronouns)
    has_female_pronoun = bool(words & female_pronouns)
    has_plural_pronoun = bool(words & plural_pronouns)
    
    # Resolve based on recent context
    if has_male_pronoun:
        for char in recent_characters:
            first_name = char.split()[0].lower()
            if first_name in gender_map["male"]:
                resolved.add(char)
                break
    
    if has_female_pronoun:
        for char in recent_characters:
            first_name = char.split()[0].lower()
            if first_name in gender_map["female"]:
                resolved.add(char)
                break
    
    if has_plural_pronoun and len(recent_characters) >= 2:
        resolved.update(list(recent_characters)[:2])
    
    return resolved

def detect_interactions_sliding_window(text, character_dict, gender_map):
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
        pronoun_characters = resolve_pronouns_heuristic(line, recent_characters, gender_map)
        
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
        if char_name.lower() == mention_lower:
            return char_name
        for alias in aliases:
            if alias == mention_lower or alias in mention_lower or mention_lower in alias:
                return char_name
    return None

def detect_interactions_fastcoref(text, character_dict, chunk_size=100):
    """Detect interactions using fastcoref for coreference resolution with chunking"""
    if not HAS_FASTCOREF:
        print("Fastcoref not installed. Falling back to sliding window.")
        return []

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
# METHOD 3: CONTEXTUAL WEIGHTING (ADVANCED)
# ============================================================================

def detect_speaker(sentence, prev_sentence, character_dict, attribution_model=None, recent_characters=None):
    """Universal speaker detection for dialogue using Heuristics or Deep Learning"""
    
    # 1. Heuristic Check (Fast)
    patterns = [
        r'said\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+said',
        r'replied\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+replied',
        r'cried\s+([A-Z][a-z]+)',
        r'([A-Z][a-z]+)\s+cried'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if match:
            name = match.group(1).lower()
            # Match to canonical name
            for canonical, aliases in character_dict.items():
                if name == canonical.lower() or name in aliases:
                    return canonical
    
    # 2. Deep Learning Attribution (If enabled and it's a quote)
    if attribution_model and ('"' in sentence or '“' in sentence) and recent_characters:
        try:
            candidate_labels = list(set(recent_characters))
            if len(candidate_labels) >= 2:
                # Combine sentence and context for better attribution
                context = sentence + " " + (prev_sentence if prev_sentence else "")
                result = attribution_model(context, candidate_labels)
                if result['scores'][0] > 0.7: # Confidence threshold
                    return result['labels'][0]
        except Exception as e:
            pass

    # 3. Fallback to previous sentence heuristic
    if ('"' in sentence or '“' in sentence) and prev_sentence:
        for pattern in patterns:
            match = re.search(pattern, prev_sentence)
            if match:
                name = match.group(1).lower()
                for canonical, aliases in character_dict.items():
                    if name == canonical.lower() or name in aliases:
                        return canonical
    return None

def get_sentiment(text):
    """Calculate sentiment polarity using TextBlob"""
    if HAS_TEXTBLOB:
        return TextBlob(text).sentiment.polarity
    return 0.0

def detect_interactions_contextual(text, character_dict, use_dl=False, decay=0.7, threshold=0.3):
    """Advanced Contextual Weighting Method with Speaker Detection and Sentiment"""
    
    if HAS_SPACY:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
    else:
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        
    attribution_model = None
    if use_dl and HAS_TRANSFORMERS:
        print("Loading Transformers Zero-Shot Classifier for Dialogue Attribution...")
        attribution_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    interactions = []
    raw_data = []
    context_weights = defaultdict(float)
    prev_sent = ""
    recent_chars = deque(maxlen=5)
    
    for i, sent in enumerate(sentences, 1):
        sent_lower = sent.lower()
        current_mentions = set()
        
        # 1. Direct Mentions
        mentions = find_character_in_text(sent, character_dict)
        for char in mentions:
            current_mentions.add(char)
            context_weights[char] = 1.0
            if char not in recent_chars:
                recent_chars.append(char)
        
        # 2. Speaker Detection (Pass recent characters for DL attribution)
        speaker = detect_speaker(sent, prev_sent, character_dict, attribution_model, list(recent_chars))
        if speaker:
            current_mentions.add(speaker)
            context_weights[speaker] = 1.0
            if speaker not in recent_chars:
                recent_chars.append(speaker)
        
        # 3. Decay
        for char in list(context_weights.keys()):
            if char not in current_mentions:
                context_weights[char] *= decay
                if context_weights[char] < 0.1:
                    del context_weights[char]
        
        # 4. Interaction Recording
        active_chars = [char for char, weight in context_weights.items() if weight > threshold]
        
        if len(active_chars) >= 2:
            sentiment = get_sentiment(sent)
            chars_sorted = sorted(active_chars)
            interactions.append(tuple([i] + chars_sorted))
            raw_data.append({
                "timestamp": i,
                "characters": chars_sorted,
                "sentiment": sentiment,
                "text": sent[:100] + "..." if len(sent) > 100 else sent
            })
        
        prev_sent = sent
        
    return interactions, raw_data

# ============================================================================
# METHOD 4: LLM (OLLAMA)
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
            timeout=300
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

def parse_llm_response(response_text):
    interactions = []
    lines = response_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove bullet prefixes
        line = re.sub(r'^[-*•]\s*', '', line)
        
        # Match "Line X:" or "X:" patterns
        match = re.match(r'(?:Line\s+)?(\d+)\s*[:)\-]\s*(.+)', line, re.IGNORECASE)
        if not match:
            continue
        
        line_num = int(match.group(1))
        mentions_str = match.group(2).strip()
        
        # Split by comma/semicolon
        raw_mentions = [m.strip() for m in re.split(r'[,;]', mentions_str) if m.strip()]

        if len(raw_mentions) >= 2:
            for i in range(len(raw_mentions)):
                raw_mentions[i] = raw_mentions[i].split("(")[0].strip()
            # Add all found characters as a group
            interactions.append((line_num, raw_mentions))

    return interactions

def detect_interactions_llm(text, paragraphs_per_chunk=7):
    """Detect interactions using local LLM, splitting by paragraphs using the TXT character list"""
    character_list = extract_characters_from_txt()
    
    lines = text.split('\n')
    
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
        
        character_list_str = "\n".join(name for name in character_list)
        prompt = f"""
        Analyze this text excerpt from Pride and Prejudice. 
        Find all interactions of 2 or more characters.
        Refer to characters by their full names from the list below:
        {character_list_str}

        Text:
        {numbered_text}

        For each interaction in this excerpt, list only the line number and characters involved.
        Group together multiple characters who interact with each other (conversation, shared action).
        Do NOT include mentions where there is no direct or indirect interaction.
        Output format: Line X: Char1, Char2, ...
        Be concise. Do not add anything apart from the required output.
        """

        response = call_ollama(prompt)
        
        if response:
            chunk_interactions = parse_llm_response(response)
            print(f"  Found {len(chunk_interactions)} interactions in this chunk")
            # Convert virtual line numbers to original line numbers and normalize names
            for virtual_line_num, chars in chunk_interactions:
                if virtual_line_num in line_mapping:
                    orig_line_num = line_mapping[virtual_line_num]
                    all_interactions.append((orig_line_num, chars))
        else:
            print(f"  Warning: No response from LLM for chunk {current_chunk}")


        print(response)

    # Convert to final format
    interactions_by_line = defaultdict(set)
    for line_num, chars in all_interactions:
        if isinstance(chars, list):
            char_group = tuple(sorted(chars))
        else:
            char_group = tuple(sorted(list(chars)))
        interactions_by_line[line_num].add(char_group)

    interactions = []
    for line_num in sorted(interactions_by_line.keys()):
        for char_group in sorted(interactions_by_line[line_num]):
            if len(char_group) >= 2:
                interactions.append((line_num,) + char_group)

    return interactions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Detect character interactions in Pride and Prejudice using different methods"
    )
    parser.add_argument(
        "method",
        nargs="?",
        default="1",
        choices=["1", "2", "3", "4", "5"],
        help="Method: 1 (sliding window), 2 (fastcoref), 3 (contextual), 4 (LLM), 5 (contextual + DL); default: 1",
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
            "3": "contextual",
            "4": "llm",
            "5": "contextual_dl"
        }[method]
        
        output_path = str(output_dir / f"interactions_{method_name}.txt")
    
    # Load universal resources
    blacklist, gender_map = load_universal_resources()
    
    # Load text and characters
    print("Loading text...")
    with open(args.text, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Loading characters...")
    if method in ['1', '2', '3', '5']:
        character_dict = extract_characters(blacklist)
    else:
        character_dict = extract_characters_from_txt(blacklist)
    print(f"Tracking {len(character_dict)} characters\n")
    
    # Run selected method
    if method == '1':
        print("=== METHOD 1: SLIDING WINDOW WITH HEURISTIC PRONOUN RESOLUTION ===\n")
        interactions = detect_interactions_sliding_window(text, character_dict, gender_map)
    
    elif method == '2':
        print("=== METHOD 2: FASTCOREF COREFERENCE RESOLUTION ===\n")
        interactions = detect_interactions_fastcoref(text, character_dict)
    
    elif method in ['3', '5']:
        use_dl = (method == '5')
        print(f"=== METHOD {method}: CONTEXTUAL WEIGHTING {'+ DL' if use_dl else ''} ===\n")
        interactions, raw_data = detect_interactions_contextual(text, character_dict, use_dl=use_dl)
        
        # Save sentiment data
        sentiment_path = output_path.replace(".txt", "_sentiment.json")
        with open(sentiment_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)
        print(f"Sentiment data saved to {sentiment_path}")
    
    elif method == '4':
        print("=== METHOD 4: LOCAL LLM (OLLAMA) ANALYSIS ===\n")
        print(f"Using model: {MODEL_NAME}")
        print(f"Using character list from: {CORRECT_CHARACTERS_PATH}")
        print(f"Make sure Ollama is running with: ollama run {MODEL_NAME}\n")
        interactions = detect_interactions_llm(text, paragraphs_per_chunk=7)
    
    # Save results
    print(f"\nFound {len(interactions)} interactions")
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for interaction in interactions:
            f.write(", ".join(map(str, interaction)) + "\n")
    
    # Generate Stats
    pair_counts = defaultdict(int)
    for inter in interactions:
        chars = inter[1:]
        for i in range(len(chars)):
            for j in range(i+1, len(chars)):
                pair = tuple(sorted([chars[i], chars[j]]))
                pair_counts[pair] += 1
    
    stats_file = output_path.replace(".txt", "_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"Total Interactions: {len(interactions)}\n\nTop 20 Pairs:\n")
        for p, c in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"{p[0]} <-> {p[1]}: {c}\n")
    print(f"Stats saved to {stats_file}")

if __name__ == "__main__":
    main()
