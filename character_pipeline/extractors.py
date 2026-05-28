import re

from character_pipeline.config import PipelineConfig
from character_pipeline.io import combine_character_lists, read_text, write_character_list


def extract_stanza(text: str) -> set[str]:
    try:
        import stanza
    except ImportError as e:
        raise SystemExit("Stanza not installed. Install with: pip install stanza") from e

    stanza.download("en", verbose=False)
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize,ner", device="cpu", verbose=False
    )
    doc = nlp(text)
    characters: set[str] = set()
    for sentence in doc.sentences:
        for ent in sentence.ents:
            if ent.type == "PERSON":
                characters.add(ent.text.strip())
    return characters


def extract_spacy(text: str, model_name: str = "en_core_web_trf") -> set[str]:
    import spacy

    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise SystemExit(
            f"SpaCy model '{model_name}' not found.\n"
            f"Install with: python -m spacy download {model_name}"
        ) from e

    doc = nlp(text)
    return {ent.text for ent in doc.ents if ent.label_ == "PERSON"}


def extract_gliner(text: str) -> set[str]:
    try:
        from gliner import GLiNER
    except ImportError as e:
        raise SystemExit("GLiNER not installed. Install with: pip install gliner") from e

    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    labels = ["person"]
    chunk_size = 1500
    all_entities = []
    chunks: list[str] = []
    current = ""
    for p in re.split(r"\n\s*\n", text):
        if not p.strip():
            continue
        if current and len(current) + len(p) + 2 > chunk_size:
            chunks.append(current)
            current = p
        else:
            current = current + "\n\n" + p if current else p
    if current:
        chunks.append(current)
    for chunk in chunks:
        entities = model.predict_entities(chunk, labels, threshold=0.5)
        all_entities.extend(entities)
    return {e["text"] for e in all_entities}


def run_stanford_ner(config: PipelineConfig) -> set[str]:
    text = read_text(config.text_path)
    characters = extract_stanza(text)
    write_character_list(config.stanford_characters, characters)
    return characters


def run_spacy_ner(config: PipelineConfig) -> set[str]:
    text = read_text(config.text_path)
    characters = extract_spacy(text, config.spacy_model)
    write_character_list(config.spacy_characters, characters)
    return characters


def run_gliner_ner(config: PipelineConfig) -> set[str]:
    text = read_text(config.text_path)
    characters = extract_gliner(text)
    write_character_list(config.gliner_characters, characters)
    return characters


def run_combine_ner(config: PipelineConfig) -> set[str]:
    combined = combine_character_lists([
        config.stanford_characters,
        config.spacy_characters,
        config.gliner_characters,
    ])
    write_character_list(config.combined_characters, combined)
    return combined
