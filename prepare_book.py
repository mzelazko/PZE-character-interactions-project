import re

# Removes preface, Gutenberg boilerplate and all illustration blocks
# potential additions: remove any '\n' and '\r' characters
def clean_justice_and_prejudice(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    start_marker = "Chapter I.]"

    end_marker = """With the Gardiners they were always on the most intimate terms. Darcy,
as well as Elizabeth, really loved them; and they were both ever
sensible of the warmest gratitude towards the persons who, by bringing
her into Derbyshire, had been the means of uniting them."""

    idx_start = text.find(start_marker)
    idx_start += len(start_marker)

    idx_end = text.find(end_marker)
    idx_end += len(end_marker)


    # replace "Chapter" with "CHAPTER" to align to other chapter names
    text_cleaned = "CHAPTER I" + text[idx_start:idx_end]

    # remove any illustration indicators
    pattern = r'\[Illustration(?::.*?)?\]+[\s\n]*'
    text_cleaned = re.sub(pattern, '', text_cleaned, flags=re.DOTALL)

    # uncomment to see what was removed with regard to illustration blocks

    # matches = list(re.finditer(pattern, text, flags=re.DOTALL))

    # for i, m in enumerate(matches, start=1):
    #     print(f"\n=== Illustration #{i} ===\n")
    #     print(m.group(0))

    output_filename = "./data/justice_and_prejudice_cleaned.txt"

    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(text_cleaned)


def main():
    text_path = "./data/pride_and_prejudice.txt"
    clean_justice_and_prejudice(text_path)

if __name__ == "__main__":
    main()