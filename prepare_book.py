import difflib
import re

TEXT_PATH = "./data/pride_and_prejudice.txt"
CLEANED_TEXT_PATH = "./data/pride_and_prejudice_cleaned.txt"
REMOVED_LOG_PATH = "./data/pride_and_prejudice_removed.log"

def remove_gutenberg_boilerplate(text: str) -> str:
    """
    Removes Gutenberg boilerplate from raw ebook text.
    Potential additions: remove any excessive newlines.
    Args:
        text: Full contents of a Project Gutenberg .txt file.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG"

    end_marker = """*** END OF THE PROJECT GUTENBERG"""

    idx_start = text.find(start_marker)
    if idx_start == -1:
        raise ValueError(f"Start marker not found: {start_marker!r}")
    idx_start = text.find("\n", idx_start) + 1  # skip " EBOOK … ***\n"

    idx_end = text.find(end_marker)
    if idx_end == -1:
        raise ValueError(f"End marker not found: {end_marker!r}")

    text_cleaned = text[idx_start:idx_end].strip()

    return text_cleaned

def _remove_illustration_blocks(text: str) -> str:
    """Drop [Illustration ...] blocks, including multi-line publisher plates."""
    lines = []
    skipping = False
    for line in text.splitlines(keepends=True):
        if not skipping:
            if re.search(r"\[Illustration", line, re.IGNORECASE):
                skipping = True
                if re.search(r"\[Illustration[^\]]*\]", line, re.IGNORECASE):
                    skipping = False
                continue
        else:
            if re.search(r"\]\s*$", line):
                skipping = False
            continue
        lines.append(line)
    return "".join(lines)


def clean_gutenberg_pride_and_prejudice(text: str) -> str:
    """
    Keeps only the Pride and Prejudice novel body: drops front matter and
    back matter via markers, removes illustration blocks, adds CHAPTER I.
    """
    start_marker = "It is a truth universally acknowledged,"
    end_marker = "had been the means of uniting them."

    idx_start = text.find(start_marker)
    if idx_start == -1:
        raise ValueError(f"Start marker not found: {start_marker!r}")

    idx_end = text.find(end_marker)
    if idx_end == -1:
        raise ValueError(f"End marker not found: {end_marker!r}")
    idx_end += len(end_marker)

    text = text[idx_start:idx_end]
    text = _remove_illustration_blocks(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return f"CHAPTER I.\n\n{text}"


def main():
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    text = remove_gutenberg_boilerplate(raw)
    ready_text = clean_gutenberg_pride_and_prejudice(text)

    with open(CLEANED_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(ready_text)

    # Save what was removed
    removed = "".join(
        line[2:]
        for line in difflib.ndiff(raw.splitlines(True), ready_text.splitlines(True))
        if line.startswith("- ")
    )
    with open(REMOVED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(removed)

if __name__ == "__main__":
    main()