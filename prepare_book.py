import re

TEXT_PATH = "./data/pride_and_prejudice.txt"
CLEANED_TEXT_PATH = "./data/pride_and_prejudice_cleaned.txt"
REMOVED_LOG_PATH = "./data/pride_and_prejudice_removed.log"


def remove_gutenberg_boilerplate(text: str) -> tuple[str, str]:
    """Return (novel region inside Gutenberg markers, text cut from raw file)."""
    start_marker = "*** START OF THE PROJECT GUTENBERG"
    end_marker = "*** END OF THE PROJECT GUTENBERG"

    idx_start = text.find(start_marker)
    if idx_start == -1:
        raise ValueError(f"Start marker not found: {start_marker!r}")
    idx_start = text.find("\n", idx_start) + 1

    idx_end = text.find(end_marker)
    if idx_end == -1:
        raise ValueError(f"End marker not found: {end_marker!r}")

    removed = text[:idx_start] + text[idx_end:]
    content = text[idx_start:idx_end].strip()
    return content, removed


def _remove_illustration_blocks(text: str) -> tuple[str, str]:
    """Drop [Illustration ...] blocks; return (kept text, removed blocks)."""
    lines = []
    removed_parts: list[str] = []
    skipping = False
    skip_buffer: list[str] = []

    for line in text.splitlines(keepends=True):
        if not skipping:
            if re.search(r"\[Illustration", line, re.IGNORECASE):
                skipping = True
                skip_buffer = [line]
                if re.search(r"\[Illustration[^\]]*\]", line, re.IGNORECASE):
                    removed_parts.append("".join(skip_buffer))
                    skipping = False
                    skip_buffer = []
                continue
        else:
            skip_buffer.append(line)
            if re.search(r"\]\s*$", line):
                removed_parts.append("".join(skip_buffer))
                skipping = False
                skip_buffer = []
            continue
        lines.append(line)

    return "".join(lines), "".join(removed_parts)


def _dewrap_paragraphs(text: str) -> str:
    """Join hard-wrapped lines within each paragraph into a single line."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged = []
    for p in paragraphs:
        if merged and p[:1].islower():
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return "\n\n".join(merged)


def clean_gutenberg_pride_and_prejudice(text: str) -> tuple[str, str]:
    """
    Keep only the Pride and Prejudice body; return (cleaned text, removed segments).
    Dewrapping reformats lines in place — that text is not counted as removed.
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

    removed = text[:idx_start]
    if text[idx_end:].strip():
        removed += text[idx_end:]

    body = text[idx_start:idx_end]
    body, illus_removed = _remove_illustration_blocks(body)
    removed += illus_removed

    body = _dewrap_paragraphs(body)
    ready_text = f"CHAPTER I.\n\n{body}"
    return ready_text, removed


def main():
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    text, removed_gutenberg = remove_gutenberg_boilerplate(raw)
    ready_text, removed_novel = clean_gutenberg_pride_and_prejudice(text)

    with open(CLEANED_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(ready_text)

    with open(REMOVED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(removed_gutenberg + removed_novel)


if __name__ == "__main__":
    main()
