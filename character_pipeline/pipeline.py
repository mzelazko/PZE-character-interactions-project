import argparse
import json
from pathlib import Path

from character_pipeline.cleaning import clean_names
from character_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from character_pipeline.extractors import (
    run_combine_ner,
    run_gliner_ner,
    run_spacy_ner,
    run_stanford_ner,
)
from character_pipeline.finalize import build_character_registry
from character_pipeline.fuzzy_clustering import write_fuzzy_aliases
from character_pipeline.io import read_character_list, write_character_list

ALL_STAGES = ("ner", "combine", "clean", "aliases", "finalize")
DEFAULT_STAGES = ("combine", "clean", "aliases", "finalize")
NER_BACKENDS = ("stanford", "spacy", "gliner", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Character extraction and alias resolution pipeline",
    )
    parser.add_argument(
        "--text-path",
        type=Path,
        default=None,
        help="Input novel text (default: pride_and_prejudice_cleaned.txt)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./results)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[*ALL_STAGES, "all"],
        default=None,
        metavar="STAGE",
        help=(
            f"Stages to run: {', '.join(ALL_STAGES)}, or all "
            f"(default: {', '.join(DEFAULT_STAGES)})"
        ),
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=NER_BACKENDS,
        default=["all"],
        help="NER backends for --stages ner (default: all)",
    )
    parser.add_argument(
        "--spacy-model",
        default=None,
        help="spaCy model for NER (default: en_core_web_trf)",
    )
    return parser.parse_args(argv)


def resolve_stages(stage_args: list[str]) -> list[str]:
    if "all" in stage_args:
        return list(ALL_STAGES)
    return stage_args


def resolve_backends(backend_args: list[str]) -> list[str]:
    if "all" in backend_args:
        return ["stanford", "spacy", "gliner"]
    return backend_args


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        text_path=args.text_path or DEFAULT_CONFIG.text_path,
        results_dir=args.results_dir or DEFAULT_CONFIG.results_dir,
    )
    if args.spacy_model:
        config.spacy_model = args.spacy_model
    return config


def run_pipeline(
    config: PipelineConfig | None = None,
    stages: list[str] | None = None,
    backends: list[str] | None = None,
) -> None:
    config = config or DEFAULT_CONFIG
    stages = stages or list(DEFAULT_STAGES)
    backends = backends or ["stanford", "spacy", "gliner"]
    config.ensure_dirs()

    print("=" * 70)
    print("CHARACTER EXTRACTION & ALIAS RESOLUTION PIPELINE")
    print("=" * 70)

    cleaned: list[str] | None = None

    if "ner" in stages:
        if not config.text_path.exists():
            raise FileNotFoundError(f"Text file not found: {config.text_path}")
        if "stanford" in backends:
            print("\n[ner] Running Stanford NER (Stanza)...")
            chars = run_stanford_ner(config)
            print(f"   Found {len(chars)} unique characters")
        if "spacy" in backends:
            print("[ner] Running spaCy NER...")
            chars = run_spacy_ner(config)
            print(f"   Found {len(chars)} unique characters")
        if "gliner" in backends:
            print("[ner] Running GLiNER...")
            chars = run_gliner_ner(config)
            print(f"   Found {len(chars)} unique characters")

    if "combine" in stages:
        print("\n[combine] Merging NER character lists...")
        combined = run_combine_ner(config)
        print(f"   Combined: {len(combined)} unique characters")

    if "clean" in stages:
        print("\n[clean] Cleaning character names...")
        raw_names = read_character_list(config.combined_characters)
        cleaned = clean_names(raw_names)
        print(
            f"   Cleaned: {len(cleaned)} names "
            f"(removed {len(raw_names) - len(cleaned)} invalid entries)"
        )
        write_character_list(config.cleaned_characters, cleaned)

    if "aliases" in stages:
        if cleaned is None:
            cleaned = read_character_list(config.cleaned_characters)
        print("\n[aliases] Fuzzy clustering (untitled names only)...")
        fuzzy = write_fuzzy_aliases(cleaned, config)
        print(f"   Seq+Fuzz: {len(fuzzy)} groups → {config.fuzzy_aliases_json}")

    if "finalize" in stages:
        if cleaned is None:
            cleaned = read_character_list(config.cleaned_characters)
        print("\n[finalize] Building character registry...")
        registry = build_character_registry(cleaned, config)
        print(f"   Saved {len(registry)} canonical characters → {config.final_characters_json}")
        print(f"   Saved aliases-only copy → {config.rule_based_aliases_json}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    if cleaned is not None:
        print(f"Extracted and cleaned {len(cleaned)} character names")
    if "finalize" in stages:
        with open(config.final_characters_json, "r", encoding="utf-8") as f:
            final_count = len(json.load(f))
        print(f"Created {final_count} canonical character groups")
        print(f"Final output: {config.final_characters_json}")
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    stage_args = args.stages if args.stages is not None else list(DEFAULT_STAGES)
    stages = resolve_stages(stage_args)
    backends = resolve_backends(args.backend)
    run_pipeline(config, stages=stages, backends=backends)


if __name__ == "__main__":
    main()
