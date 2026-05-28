#!/usr/bin/env python3
"""
Backward-compatible entry point for the character extraction pipeline.

Implementation lives in the character_pipeline package.
"""
from pathlib import Path

from character_pipeline.cleaning import clean_names, normalize_name
from character_pipeline.config import (
    CHAPTER_RE,
    DEFAULT_CONFIG,
    MALE_TITLES,
    NAME_PARTICLES,
    NICKNAMES,
    PLACE_NAMES,
    TITLES,
    PipelineConfig,
)
from character_pipeline.extractors import (
    run_combine_ner,
    run_gliner_ner,
    run_spacy_ner,
    run_stanford_ner,
)
from character_pipeline.finalize import build_character_registry
from character_pipeline.fuzzy_clustering import write_fuzzy_aliases
from character_pipeline.io import (
    combine_character_lists,
    sort_by_alias_count,
    sort_by_position_count,
    write_character_list,
)
from character_pipeline.name_parsing import ignore_words, parse_name
from character_pipeline.pipeline import main, run_pipeline
from character_pipeline.rule_clustering import cluster_rule_based
from character_pipeline.similarity import fuzzy_similarity, seq_similarity

_cfg = DEFAULT_CONFIG
TEXT_PATH = str(_cfg.text_path)
RESULTS_DIR = _cfg.results_dir
EXTRACTION_DIR = _cfg.extraction_dir
ALIASES_DIR = _cfg.aliases_dir
STANFORD_CHARACTER_OUTPUT = _cfg.stanford_characters
SPACY_CHARACTER_OUTPUT = _cfg.spacy_characters
GLINER_CHARACTER_OUTPUT = _cfg.gliner_characters
COMBINED_CHARACTER_OUTPUT = _cfg.combined_characters
CLEANED_FILE = _cfg.cleaned_characters
ALIASES_FILE = _cfg.fuzzy_aliases_json
FINAL_JSON = _cfg.final_characters_json
RULE_BASED_ALIASES = _cfg.rule_based_aliases_json
SIMILARITY_THRESHOLD = _cfg.similarity_threshold

_ignore_words = ignore_words
cluster_names_greedy = cluster_rule_based


def _config_for_text(text_path) -> PipelineConfig:
    return PipelineConfig(
        text_path=Path(text_path),
        results_dir=_cfg.results_dir,
        similarity_threshold=_cfg.similarity_threshold,
        fuzzy_threshold=_cfg.fuzzy_threshold,
        spacy_model=_cfg.spacy_model,
        scan_spacy_model=_cfg.scan_spacy_model,
    )


def ner_stanford(text_path=TEXT_PATH):
    return run_stanford_ner(_config_for_text(text_path))


def ner_spacy(text_path=TEXT_PATH, model_name=None):
    cfg = _config_for_text(text_path)
    if model_name:
        cfg.spacy_model = model_name
    return run_spacy_ner(cfg)


def ner_gliner(text_path=TEXT_PATH):
    return run_gliner_ner(_config_for_text(text_path))


def combine_ner_results(stanford_file, spacy_file, gliner_file, output_file):
    combined = combine_character_lists([
        Path(stanford_file),
        Path(spacy_file),
        Path(gliner_file),
    ])
    write_character_list(Path(output_file), combined)
    return combined


def generate_alias_files(names):
    return write_fuzzy_aliases(names, DEFAULT_CONFIG)


def finalize_characters(cleaned_names):
    return build_character_registry(cleaned_names, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
