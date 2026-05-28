from character_pipeline.config import PipelineConfig
from character_pipeline.io import sort_by_alias_count, write_json
from character_pipeline.name_parsing import parse_name
from character_pipeline.similarity import seq_fuzz_similarity


def cluster_untitled_fuzzy(
    names: list[str],
    threshold: float = 0.75,
) -> dict[str, list[str]]:
    """Seq+Fuzz graph clustering on names without a title."""
    untitled = [n for n in names if parse_name(n).title == ""]
    remaining = set(untitled)
    clusters: list[list[str]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            cur = stack.pop()
            for other in list(remaining):
                if seq_fuzz_similarity(cur, other) >= threshold:
                    remaining.remove(other)
                    component.add(other)
                    stack.append(other)
        clusters.append(sorted(component))
    return sort_by_alias_count({c[0]: c for c in clusters})


def write_fuzzy_aliases(
    names: list[str],
    config: PipelineConfig,
) -> dict[str, list[str]]:
    result = cluster_untitled_fuzzy(names, threshold=config.fuzzy_threshold)
    write_json(config.fuzzy_aliases_json, result)
    return result
