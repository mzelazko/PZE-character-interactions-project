import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Full Character Interaction Analysis Pipeline")
    parser.add_argument("-t", "--text", default="data/pride_and_prejudice_cleaned.txt", help="Input text file")
    parser.add_argument("-c", "--chars", default="results/character_extraction/final_characters.json", help="Characters JSON file")
    parser.add_argument("-m", "--method", default="5", choices=["1", "2", "3", "5"], help="Method to use for final analysis")
    
    args = parser.parse_args()
    
    print("=== Step 1: Interaction Detection ===")
    out_txt = "results/interactions/final_interactions.txt"
    os.system(f"python3 interactions.py {args.method} -t {args.text} -c {args.chars} -o {out_txt}")
    
    print("\n=== Step 2: Network Visualization ===")
    sentiment_json = "results/interactions/sentiment_contextual_dl.json" if args.method == "5" else "results/interactions/sentiment_contextual.json"
    os.system(f"python3 visualize_network.py -i {sentiment_json} -o results/interactions/final_network.html")
    
    print("\n=== Step 3: Temporal Analysis ===")
    os.system(f"python3 temporal_analysis.py -i {sentiment_json} -o results/interactions/ -w 200")
    
    print("\n=== Step 4: Method Comparison ===")
    os.system(f"python3 evaluate_methods.py -t data/pride_and_prejudice_sample.txt -c {args.chars} -o results/method_comparison_report.md")
    
    print("\n=== Pipeline Completed Successfully ===")
    print("Results available in results/interactions/")

if __name__ == "__main__":
    main()
