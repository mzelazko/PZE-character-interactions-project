import json
import os
import argparse
from collections import defaultdict

def evaluate_methods(text_path, char_path, output_report):
    # This script will run all methods and compare their top pairs
    # as a proxy for performance, since we don't have a gold standard yet.
    
    methods = {
        "1": "Sliding Window",
        "2": "FastCoref Hybrid",
        "3": "Contextual",
        "5": "Contextual + DL Attribution"
    }
    
    results = {}
    
    for m_id, m_name in methods.items():
        print(f"Running Method {m_id}: {m_name}...")
        out_file = f"results/eval_method_{m_id}.txt"
        cmd = f"python3 interactions.py {m_id} -t {text_path} -c {char_path} -o {out_file} > /dev/null 2>&1"
        os.system(cmd)
        
        # Parse stats
        stats_file = out_file.replace(".txt", "_stats.txt")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                lines = f.readlines()
                total = int(lines[0].split(":")[1].strip())
                top_pairs = []
                for line in lines[3:13]: # Top 10
                    if "<->" in line:
                        pair, count = line.split(":")
                        top_pairs.append((pair.strip(), int(count.strip())))
                results[m_name] = {"total": total, "top_pairs": top_pairs}

    # Generate Report
    with open(output_report, 'w') as f:
        f.write("# Method Comparison Report\n\n")
        f.write("| Method | Total Interactions | Top Pair | Top Pair Count |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for name, data in results.items():
            top_p = data['top_pairs'][0][0] if data['top_pairs'] else "N/A"
            top_c = data['top_pairs'][0][1] if data['top_pairs'] else 0
            f.write(f"| {name} | {data['total']} | {top_p} | {top_c} |\n")
        
        f.write("\n## Detailed Top 10 Pairs Comparison\n\n")
        for name, data in results.items():
            f.write(f"### {name}\n")
            for pair, count in data['top_pairs']:
                f.write(f"- {pair}: {count}\n")
            f.write("\n")

    print(f"Evaluation report saved to {output_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Interaction Detection Methods")
    parser.add_argument("-t", "--text", required=True, help="Input text file")
    parser.add_argument("-c", "--chars", required=True, help="Characters JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output report file")
    
    args = parser.parse_args()
    evaluate_methods(args.text, args.chars, args.output)
