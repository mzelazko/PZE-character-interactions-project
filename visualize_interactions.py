import matplotlib.pyplot as plt
import pandas as pd

def main():
    stats_path = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/full_novel_stats.txt"
    
    pairs = []
    counts = []
    
    with open(stats_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if "<->" in line:
                parts = line.split(":")
                pair = parts[0].strip()
                count = int(parts[1].strip())
                pairs.append(pair)
                counts.append(count)
    
    # Take top 15 for better visualization
    df = pd.DataFrame({"Pair": pairs[:15], "Interactions": counts[:15]})
    df = df.sort_values(by="Interactions", ascending=True)
    
    plt.figure(figsize=(12, 8))
    plt.barh(df["Pair"], df["Interactions"], color='skyblue')
    plt.xlabel("Number of Interactions")
    plt.title("Top 15 Character Interactions in Pride and Prejudice")
    plt.tight_layout()
    
    output_img = "/home/ubuntu/pze_project/PZE-character-interactions-project-main/top_interactions.png"
    plt.savefig(output_img)
    print(f"Visualization saved to {output_img}")

if __name__ == "__main__":
    main()
