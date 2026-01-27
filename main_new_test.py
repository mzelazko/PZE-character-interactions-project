import os
import sys
from prepare_book import clean_pride_and_prejudice
from unified_interactions import InteractionAnalyzer

# Paths
RAW_TEXT = "./data/pride_and_prejudice.txt"
CLEANED_TEXT = "./data/pride_and_prejudice_cleaned.txt"
CHAR_DICT = "./final_characters.json"
OUTPUT_JSON = "./results/final_interactions_full.json"
OUTPUT_STATS = "./results/final_stats_full.txt"

def main():
    # 1. Ensure directories exist
    os.makedirs("./results", exist_ok=True)
    
    # 2. Clean the book if needed
    if not os.path.exists(CLEANED_TEXT):
        print("Cleaning book text...")
        clean_pride_and_prejudice(RAW_TEXT)
    
    # 3. Initialize Analyzer
    # Set use_coref=True for best results, False for faster processing
    analyzer = InteractionAnalyzer(CHAR_DICT, use_coref=True)
    
    # 4. Run Full Analysis
    print("Starting full novel analysis...")
    analyzer.process_book(CLEANED_TEXT, OUTPUT_JSON, OUTPUT_STATS)
    
    print("\n" + "="*30)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final interactions saved to: {OUTPUT_JSON}")
    print(f"Statistics summary saved to: {OUTPUT_STATS}")
    print("="*30)

if __name__ == "__main__":
    main()
