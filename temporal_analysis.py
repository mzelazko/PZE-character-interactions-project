import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def analyze_temporal_arcs(sentiment_json_path, output_dir, window_size=200):
    if not os.path.exists(sentiment_json_path):
        print(f"Error: {sentiment_json_path} not found.")
        return

    with open(sentiment_json_path, 'r', encoding='utf-8') as f:
        interactions = json.load(f)

    # Convert to DataFrame
    data = []
    for inter in interactions:
        chars = sorted(inter['characters'])
        if len(chars) >= 2:
            for i in range(len(chars)):
                for j in range(i + 1, len(chars)):
                    pair = f"{chars[i]} - {chars[j]}"
                    data.append({
                        'timestamp': inter['timestamp'],
                        'pair': pair,
                        'sentiment': inter.get('sentiment', 0)
                    })
    
    df = pd.DataFrame(data)
    
    # Get top pairs to plot
    top_pairs = df['pair'].value_counts().head(5).index.tolist()
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for pair in top_pairs:
        pair_df = df[df['pair'] == pair].sort_values('timestamp')
        # Rolling average for sentiment
        pair_df['rolling_sentiment'] = pair_df['sentiment'].rolling(window=window_size, min_periods=1).mean()
        plt.plot(pair_df['timestamp'], pair_df['rolling_sentiment'], label=pair)
    
    plt.title(f"Sentiment Arcs of Top Character Pairs (Window Size: {window_size})")
    plt.xlabel("Sentence Index (Time)")
    plt.ylabel("Average Sentiment")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "sentiment_arcs.png")
    plt.savefig(plot_path)
    print(f"Sentiment arcs plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Temporal Sentiment Arcs")
    parser.add_argument("-i", "--input", required=True, help="Input sentiment JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-w", "--window", type=int, default=100, help="Rolling window size")
    
    args = parser.parse_args()
    analyze_temporal_arcs(args.input, args.output, args.window)
