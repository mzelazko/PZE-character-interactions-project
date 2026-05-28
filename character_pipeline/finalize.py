import re
from collections import defaultdict

from character_pipeline.config import PipelineConfig
from character_pipeline.io import sort_by_alias_count, sort_by_position_count, write_json
from character_pipeline.rule_clustering import cluster_rule_based
from character_pipeline.types import CharacterRecord


def compute_sentence_occurrences(
    book_text: str,
    names: list[str],
    model_name: str = "en_core_web_sm",
) -> dict[str, list[int]]:
    """Find 1-based sentence indices where each name appears (word-boundary match)."""
    import spacy

    nlp = spacy.load(model_name)
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")

    doc = nlp(book_text)
    sentences = [sent.text for sent in doc.sents]

    try:
        from tqdm import tqdm

        iterator = enumerate(tqdm(sentences, desc="   Scanning sentences"), start=1)
    except Exception:
        iterator = enumerate(sentences, start=1)

    occ: dict[str, set[int]] = defaultdict(set)
    regexes = {
        name: re.compile(r"\b" + re.escape(name) + r"\b", flags=re.IGNORECASE)
        for name in names
    }
    for i, sent in iterator:
        for name, rx in regexes.items():
            if rx.search(sent):
                occ[name].add(i)
    return {k: sorted(v) for k, v in occ.items()}


# Backward-compatible alias
find_missing_occurrences_by_scanning = compute_sentence_occurrences


def build_character_registry(
    cleaned_names: list[str],
    config: PipelineConfig,
) -> dict[str, CharacterRecord]:
    """Cluster aliases, attach sentence positions, write final JSON outputs."""
    if not config.text_path.exists():
        raise FileNotFoundError(f"Book text not found for position scan: {config.text_path}")

    book_text = config.text_path.read_text(encoding="utf-8")
    occ = compute_sentence_occurrences(
        book_text, cleaned_names, config.scan_spacy_model
    )
    counts = {name: len(occ.get(name, [])) for name in cleaned_names}
    _mapping, groups = cluster_rule_based(cleaned_names, counts)

    registry: dict[str, CharacterRecord] = {}
    for canon, aliases in groups.items():
        positions: set[int] = set()
        for a in aliases:
            positions.update(occ.get(a, []))
        registry[canon] = {
            "aliases": sorted(aliases),
            "positions": sorted(positions),
        }

    aliases_only = sort_by_alias_count(
        {canon: {"aliases": info["aliases"]} for canon, info in registry.items()}
    )
    write_json(config.rule_based_aliases_json, aliases_only)

    registry = sort_by_position_count(registry)
    write_json(config.final_characters_json, registry)

    return registry


# Backward-compatible alias
finalize_characters = build_character_registry
