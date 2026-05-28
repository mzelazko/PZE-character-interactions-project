"""Character extraction and alias resolution pipeline."""

from character_pipeline.cleaning import clean_names, normalize_name
from character_pipeline.config import (
    DEFAULT_CONFIG,
    MALE_TITLES,
    NICKNAMES,
    PLACE_NAMES,
    TITLES,
    PipelineConfig,
)
from character_pipeline.finalize import (
    build_character_registry,
    compute_sentence_occurrences,
    finalize_characters,
    find_missing_occurrences_by_scanning,
)
from character_pipeline.fuzzy_clustering import cluster_untitled_fuzzy, write_fuzzy_aliases
from character_pipeline.name_parsing import ParsedName, ignore_words, parse_name
from character_pipeline.pipeline import main, run_pipeline
from character_pipeline.rule_clustering import cluster_rule_based
from character_pipeline.similarity import fuzzy_similarity, seq_similarity

__all__ = [
    "DEFAULT_CONFIG",
    "MALE_TITLES",
    "NICKNAMES",
    "PLACE_NAMES",
    "TITLES",
    "ParsedName",
    "PipelineConfig",
    "clean_names",
    "cluster_rule_based",
    "build_character_registry",
    "cluster_untitled_fuzzy",
    "compute_sentence_occurrences",
    "finalize_characters",
    "find_missing_occurrences_by_scanning",
    "fuzzy_similarity",
    "ignore_words",
    "main",
    "normalize_name",
    "parse_name",
    "run_pipeline",
    "seq_similarity",
    "write_fuzzy_aliases",
]
