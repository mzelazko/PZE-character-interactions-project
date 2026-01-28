import json
import networkx as nx
from pyvis.network import Network
import os
import argparse

def create_visual_network(sentiment_json_path, output_html_path):
    if not os.path.exists(sentiment_json_path):
        print(f"Error: {sentiment_json_path} not found.")
        return

    with open(sentiment_json_path, 'r', encoding='utf-8') as f:
        interactions = json.load(f)

    G = nx.Graph()

    # Aggregate interactions into edges
    edges = {}
    for inter in interactions:
        chars = inter['characters']
        sentiment = inter.get('sentiment', 0)
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                pair = tuple(sorted([chars[i], chars[j]]))
                if pair not in edges:
                    edges[pair] = {'weight': 0, 'sentiment_sum': 0}
                edges[pair]['weight'] += 1
                edges[pair]['sentiment_sum'] += sentiment

    # Add edges to graph
    for pair, data in edges.items():
        avg_sentiment = data['sentiment_sum'] / data['weight']
        # Color based on sentiment: Green for positive, Red for negative, Gray for neutral
        color = '#808080' # Gray
        if avg_sentiment > 0.1:
            color = '#00FF00' # Green
        elif avg_sentiment < -0.1:
            color = '#FF0000' # Red
            
        G.add_edge(pair[0], pair[1], 
                   weight=data['weight'], 
                   title=f"Interactions: {data['weight']}\nAvg Sentiment: {avg_sentiment:.2f}",
                   value=data['weight'],
                   color=color)

    # Create PyVis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.from_nx(G)
    
    # Set physics for better layout
    net.toggle_physics(True)
    
    # Save the network
    net.save_graph(output_html_path)
    print(f"Network visualization saved to {output_html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Character Interaction Network")
    parser.add_argument("-i", "--input", required=True, help="Input sentiment JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output HTML file")
    
    args = parser.parse_args()
    create_visual_network(args.input, args.output)
