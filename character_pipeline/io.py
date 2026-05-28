import json
from pathlib import Path


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_character_list(path: Path, characters: set[str] | list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted(characters):
            f.write(name + "\n")


def read_character_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Character list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def combine_character_lists(paths: list[Path]) -> set[str]:
    sets: list[set[str]] = []
    for file_path in paths:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t")[0] for line in f if line.strip()]
            sets.append(set(lines))
    return set().union(*sets)


def write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sort_mapping(mapping: dict, key_fn) -> dict:
    return dict(sorted(mapping.items(), key=lambda kv: (-key_fn(kv[1]), kv[0])))


def alias_count(entry: list | dict) -> int:
    if isinstance(entry, list):
        return len(entry)
    return len(entry.get("aliases", []))


def position_count(entry: dict) -> int:
    return len(entry.get("positions", []))


def sort_by_alias_count(mapping: dict) -> dict:
    return sort_mapping(mapping, alias_count)


def sort_by_position_count(mapping: dict) -> dict:
    return sort_mapping(mapping, position_count)
