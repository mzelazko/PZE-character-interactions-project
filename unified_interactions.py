import json
import os
import re
import spacy
import spacy.cli
from collections import defaultdict, deque
from fastcoref import FCoref
from textblob import TextBlob

class InteractionAnalyzer:
    def __init__(self, char_dict_path, use_coref=True, device='cpu'):
        self.use_coref = use_coref
        self.char_dict = self._load_char_dict(char_dict_path)
        
        # List of generic terms to filter out from character detection
        self.generic_filters = {
            'man', 'woman', 'lady', 'sir', 'miss', 'mrs', 'mr', 
            'longbourn', 'pemberley', 'netherfield', 'meryton', 
            'hertfordshire', 'derbyshire', 'london', 'england',
            'woods', 'park', 'house', 'hall', 'street', 'colonel',
            'captain', 'general', 'lieutenant', 'major', 'sergeant',
            'master', 'mistress', 'servant', 'gentleman', 'gentlemen',
            'friend', 'acquaintance', 'cousin', 'aunt', 'uncle', 'sister',
            'brother', 'father', 'mother', 'daughter', 'son', 'family'
        }
        
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
                
                # Filter out generic antecedents
                if antecedent.lower().strip() in self.generic_filters:
                    continue
                
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

    def detect_speaker(self, sentence, prev_sentence):
        """Heuristic speaker detection for dialogue."""
        # Look for patterns like: "..." said Elizabeth or Elizabeth said, "..."
        patterns = [
            r'said\s+([A-Z][a-z]+)',
            r'([A-Z][a-z]+)\s+said',
            r'replied\s+([A-Z][a-z]+)',
            r'([A-Z][a-z]+)\s+replied',
            r'cried\s+([A-Z][a-z]+)',
            r'([A-Z][a-z]+)\s+cried'
        ]
        
        # Check current sentence
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                name = match.group(1)
                for char_name, aliases in self.char_dict.items():
                    if name.lower() in aliases:
                        return char_name
        
        # Check previous sentence if current has dialogue but no speaker
        if ('"' in sentence or '“' in sentence) and prev_sentence:
            for pattern in patterns:
                match = re.search(pattern, prev_sentence)
                if match:
                    name = match.group(1)
                    for char_name, aliases in self.char_dict.items():
                        if name.lower() in aliases:
                            return char_name
                            
        return None

    def get_sentiment(self, text):
        """Calculate sentiment polarity using TextBlob."""
        return TextBlob(text).sentiment.polarity

    def detect_interactions(self, sentences, decay=0.7, threshold=0.3):
        """Detect interactions using contextual weighting, speaker detection, and sentiment."""
        interactions = []
        context_weights = defaultdict(float)
        prev_sent = ""
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            current_mentions = set()
            
            # 1. Direct Mention Detection with Filtering
            for char_name, aliases in self.char_dict.items():
                for alias in aliases:
                    # Skip generic aliases
                    if alias in self.generic_filters:
                        continue
                    if re.search(r'\b' + re.escape(alias) + r'\b', sent_lower):
                        current_mentions.add(char_name)
                        context_weights[char_name] = 1.0
                        break
            
            # 2. Speaker Detection
            speaker = self.detect_speaker(sent, prev_sent)
            if speaker:
                current_mentions.add(speaker)
                context_weights[speaker] = 1.0
            
            # 3. Decay
            for char in list(context_weights.keys()):
                if char not in current_mentions:
                    context_weights[char] *= decay
                    if context_weights[char] < 0.1:
                        del context_weights[char]
            
            # 4. Interaction Recording
            active_chars = [char for char, weight in context_weights.items() if weight > threshold]
            
            if len(active_chars) >= 2:
                sentiment_polarity = self.get_sentiment(sent)
                interactions.append({
                    "sentence_index": i,
                    "characters": sorted(active_chars),
                    "text": sent,
                    "speaker": speaker,
                    "sentiment": sentiment_polarity
                })
            
            prev_sent = sent
            
        return interactions

    def process_book(self, text_path, output_json, output_stats):
        """Process the entire book chapter by chapter."""
        with open(text_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        chapters = re.split(r'(CHAPTER [IVXLCDM]+)', full_text)
        all_interactions = []
        
        chapters = [c for c in chapters if c.strip()]
        
        print(f"Processing chapters with Sentiment Analysis...")
        
        for i in range(0, len(chapters), 2):
            if i + 1 >= len(chapters):
                break
            chapter_title = chapters[i]
            chapter_content = chapters[i+1]
            full_chapter = chapter_title + chapter_content
            
            print(f"Analyzing {chapter_title.strip()}...")
            
            processed_text = self.resolve_coreferences(full_chapter)
            doc = self.nlp(processed_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            chapter_inters = self.detect_interactions(sentences)
            for inter in chapter_inters:
                inter['chapter'] = chapter_title.strip()
                all_interactions.append(inter)
                
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_interactions, f, indent=2)
            
        # Generate stats
        pair_counts = defaultdict(int)
        sentiment_sums = defaultdict(float)
        
        for inter in all_interactions:
            chars = inter['characters']
            sentiment = inter['sentiment']
            for idx in range(len(chars)):
                for jdx in range(idx + 1, len(chars)):
                    pair = tuple(sorted([chars[idx], chars[jdx]]))
                    pair_counts[pair] += 1
                    sentiment_sums[pair] += sentiment
                    
        # Calculate average sentiment
        avg_sentiments = {pair: sentiment_sums[pair] / pair_counts[pair] for pair in pair_counts}
        
        # Sort by count and sentiment
        sorted_by_count = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_by_sentiment = sorted(avg_sentiments.items(), key=lambda x: x[1], reverse=True)
        
        with open(output_stats, "w", encoding="utf-8") as f:
            f.write(f"Total interactions (Sentiment-Aware): {len(all_interactions)}\n\n")
            f.write("Top Character Pairs (by Count):\n")
            for pair, count in sorted_by_count[:50]:
                f.write(f"{pair[0]} <-> {pair[1]}: {count} (Avg Sentiment: {avg_sentiments[pair]:.4f})\n")
            
            f.write("\nMost Positive Relationships (Top 20):\n")
            for pair, avg_sentiment in sorted_by_sentiment[:20]:
                f.write(f"{pair[0]} <-> {pair[1]}: {avg_sentiment:.4f} (Count: {pair_counts[pair]})\n")

            f.write("\nMost Negative Relationships (Top 20):\n")
            for pair, avg_sentiment in sorted_by_sentiment[-20:]:
                f.write(f"{pair[0]} <-> {pair[1]}: {avg_sentiment:.4f} (Count: {pair_counts[pair]})\n")
        
        print(f"Sentiment-aware analysis complete. Results: {output_json}, Stats: {output_stats}")
        return all_interactions
