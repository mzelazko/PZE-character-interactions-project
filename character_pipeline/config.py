import re
from dataclasses import dataclass, field
from pathlib import Path

CHAPTER_RE = re.compile(r"^(?:CHAPTER|Chapter|chapter)\s+[IVXLCDM\d]+\.?$")

NAME_PARTICLES = {"de", "del", "da", "di", "la", "le", "van", "von", "der", "den", "of", "the", "y"}

TITLES = {
    "mr", "mrs", "miss", "ms", "sir", "lady", "lord",
    "dr", "rev", "esq", "esquire",
    "colonel", "captain", "major", "general",
    "honourable", "honored", "honoured", "right",
}
MALE_TITLES = {"mr", "sir", "lord", "colonel", "captain", "major", "general"}

PLACE_NAMES = {
    "pemberley", "longbourn", "netherfield", "meryton", "rosings",
    "hunsford", "lambton", "brighton", "kent", "bakewell", "matlock",
    "dovedale", "bromley", "ashworth", "grantley", "hatfield",
    "hertfordshire", "ramsgate", "clapham", "cheapside",
    "lucas lodge", "pemberley woods", "haye park", "purvis lodge",
    "hunsford parsonage", "rosings park",
}

NICKNAMES = {
    "lizzy": "elizabeth",
    "lizzie": "elizabeth",
    "eliza": "elizabeth",
}


@dataclass
class PipelineConfig:
    text_path: Path = field(default_factory=lambda: Path("./data/pride_and_prejudice_cleaned.txt"))
    results_dir: Path = field(default_factory=lambda: Path("./results"))
    similarity_threshold: float = 0.8
    fuzzy_threshold: float = 0.75
    spacy_model: str = "en_core_web_trf"
    scan_spacy_model: str = "en_core_web_sm"

    @property
    def extraction_dir(self) -> Path:
        return self.results_dir / "character_extraction"

    @property
    def aliases_dir(self) -> Path:
        return self.results_dir / "aliases"

    @property
    def stanford_characters(self) -> Path:
        return self.extraction_dir / "stanford_characters.txt"

    @property
    def spacy_characters(self) -> Path:
        return self.extraction_dir / "spacy_characters.txt"

    @property
    def gliner_characters(self) -> Path:
        return self.extraction_dir / "gliner_characters.txt"

    @property
    def combined_characters(self) -> Path:
        return self.extraction_dir / "combined_characters.txt"

    @property
    def cleaned_characters(self) -> Path:
        return self.extraction_dir / "cleaned_characters.txt"

    @property
    def fuzzy_aliases_json(self) -> Path:
        return self.aliases_dir / "distance_metrics_aliases.json"

    @property
    def rule_based_aliases_json(self) -> Path:
        return self.aliases_dir / "rule_based_aliases.json"

    @property
    def final_characters_json(self) -> Path:
        return self.results_dir / "final_characters.json"

    def ensure_dirs(self) -> None:
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        self.aliases_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = PipelineConfig()
