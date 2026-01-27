import json
import os
import re
import spacy
import spacy.cli
from collections import defaultdict, deque
from fastcoref import FCoref

class InteractionAnalyzer:
    def __init__(self, char_dict_path, use_coref=True, device='cpu'):
        self.use_coref = use_coref
        self.char_dict = self._load_char_dict(char_dict_path)
        
        print("Loading SpaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            print("Downloading SpaCy model 'en_core_web_lg'...")
            spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")
            
        if self.use_coref:
            print("Loading fastcoref model...")
            self.coref_model = FCoref(device=device)
        else:
            self.coref_model = None

    def _load_char_dict(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {name: [a.lower() for a in info['aliases']] for name, info in data.items()}

    def resolve_coreferences(self, text):
        """Resolve coreferences using fastcoref."""
        if not self.coref_model:
            return text
        try:
            preds = self.coref_model.predict(texts=[text])
            result = preds[0]
            clusters = result.clusters
            char_map = result.char_map
            text_list = list(text)
            replacements = []
            
            for cluster in clusters:
                antecedent_token_span = cluster[0]
                if antecedent_token_span not in char_map:
                    continue
                _, (ant_start, ant_end) = char_map[antecedent_token_span]
                antecedent = text[ant_start:ant_end]
                
                for mention_token_span in cluster[1:]:
                    if mention_token_span in char_map:
                        _, (m_start, m_end) = char_map[mention_token_span]
                        replacements.append({'start': m_start, 'end': m_end, 'replacement': antecedent})

            replacements.sort(key=lambda x: x['start'], reverse=True)
            for rep in replacements:
                text_list[rep['start']:rep['end']] = list(rep['replacement'])
            return "".join(text_list)
        except Exception as e:
            print(f"Coreference resolution warning: {e}")
            return text

    def detect_interactions(self, sentences, decay=0.7, threshold=0.3):
        """Detect interactions using contextual weighting."""
        interactions = []
        context_weights = defaultdict(float)
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            current_mentions = set()
            
            for char_name, aliases in self.char_dict.items():
                for alias in aliases:
                    if re.search(r'\b' + re.escape(alias) + r'\b', sent_lower):
                        current_mentions.add(char_name)
                        context_weights[char_name] = 1.0
                        break
            
            # Decay
            for char in list(context_weights.keys()):
                if char not in current_mentions:
                    context_weights[char] *= decay
                    if context_weights[char] < 0.1:
                        del context_weights[char]
            
            active_chars = [char for char, weight in context_weights.items() if weight > threshold]
            
            if len(active_chars) >= 2:
                interactions.append({
                    "sentence_index": i,
                    "characters": sorted(active_chars),
                    "text": sent
                })
        return interactions

    def process_book(self, text_path, output_json, output_stats):
        """Process the entire book chapter by chapter."""
        with open(text_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        chapters = re.split(r'(CHAPTER [IVXLCDM]+)', full_text)
        all_interactions = []
        
        # Filter out empty strings from split
        chapters = [c for c in chapters if c.strip()]
        
        print(f"Processing chapters...")
        
        # Iterate through pairs of (Chapter Title, Content)
        for i in range(0, len(chapters), 2):
            if i + 1 >= len(chapters):
                break
            chapter_title = chapters[i]
            chapter_content = chapters[i+1]
            full_chapter = chapter_title + chapter_content
            
            print(f"Analyzing {chapter_title.strip()}...")
            
            # 1. Coref
            processed_text = self.resolve_coreferences(full_chapter)
            
            # 2. Sentences
            doc = self.nlp(processed_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            # 3. Interactions
            chapter_inters = self.detect_interactions(sentences)
            for inter in chapter_inters:
                inter['chapter'] = chapter_title.strip()
                all_interactions.append(inter)
                
        # Save results
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_interactions, f, indent=2)
            
        # Generate stats
        pair_counts = defaultdict(int)
        for inter in all_interactions:
            chars = inter['characters']
            for idx in range(len(chars)):
                for jdx in range(idx + 1, len(chars)):
                    pair = tuple(sorted([chars[idx], chars[jdx]]))
                    pair_counts[pair] += 1
                    
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        with open(output_stats, "w", encoding="utf-8") as f:
            f.write(f"Total interactions: {len(all_interactions)}\n\n")
            for pair, count in sorted_pairs[:50]:
                f.write(f"{pair[0]} <-> {pair[1]}: {count}\n")
        
        print(f"Analysis complete. Results: {output_json}, Stats: {output_stats}")
        return all_interactions
