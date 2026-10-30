"""Mine Europe PMC abstracts for HER2-related chemical compounds.

Pipeline:
  1. Query the Europe PMC REST API and fetch the top N abstracts.
  2. Run biomedical NER (chemical + gene/protein models) on each sentence.
  3. Keep abstracts that mention HER2/ERBB2 and at least one chemical.
  4. Export (chemical, context sentence, PMID) rows to data/mined_compounds.csv.

Note on model choice: 'd4data/biomedical-ner-all' is trained on clinical case
reports (MACCROBAT) and has no Chemical/Protein labels, so we use the BENT
PubMedBERT models, which are trained specifically on chemical and gene/protein
entities in biomedical literature.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Fail fast (and let huggingface_hub retry) instead of hanging forever on a
# stalled model download. Must be set before transformers is imported.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
# The Xet transfer backend was observed to stall indefinitely at 0 bytes on
# some networks; plain HTTP downloads are slower to start but reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import requests
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "mined_compounds.csv"

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SEARCH_QUERY = 'HER2 AND (phytochemical OR "plant extract" OR inhibitor)'
# Restrict to PubMed records that actually have an abstract (some hits are
# full-text-only PMC records without a PMID or abstract).
QUERY_FILTERS = "HAS_ABSTRACT:y AND SRC:MED"
MAX_RESULTS = 50

CHEMICAL_MODEL_ID = "pruas/BENT-PubMedBERT-NER-Chemical"
GENE_MODEL_ID = "pruas/BENT-PubMedBERT-NER-Gene"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models"


def resolve_model(model_id: str) -> str:
    """Prefer a local copy in models/<name> (with weights) over the HF Hub id."""
    local = LOCAL_MODELS_DIR / model_id.split("/")[-1]
    has_weights = any((local / f).exists() for f in ("pytorch_model.bin", "model.safetensors"))
    return str(local) if has_weights else model_id


CHEMICAL_MODEL = resolve_model(CHEMICAL_MODEL_ID)
GENE_MODEL = resolve_model(GENE_MODEL_ID)

HER2_PATTERN = re.compile(r"\b(HER-?2(/neu)?|ERBB-?2|c-?erbB-?2|neu)\b", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
MIN_ENTITY_SCORE = 0.70


# --------------------------------------------------------------------------- #
# Step 1: Europe PMC retrieval
# --------------------------------------------------------------------------- #
def fetch_abstracts(query: str, max_results: int) -> list[dict]:
    """Return up to `max_results` records with pmid, title and abstract."""
    params = {
        "query": f"({query}) AND {QUERY_FILTERS}",
        "format": "json",
        "resultType": "core",  # 'core' includes abstractText
        "pageSize": min(max_results, 1000),
        "cursorMark": "*",
    }
    records: list[dict] = []
    while len(records) < max_results:
        for attempt in range(3):
            try:
                resp = requests.get(EUROPE_PMC_URL, params=params, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    raise
                print(f"  Request failed ({exc}); retrying...")
                time.sleep(2 * (attempt + 1))
        payload = resp.json()
        results = payload.get("resultList", {}).get("result", [])
        if not results:
            break
        for item in results:
            abstract = item.get("abstractText")
            pmid = item.get("pmid")
            if not abstract or not pmid:
                continue
            records.append(
                {
                    "pmid": pmid,
                    "title": item.get("title", ""),
                    "abstract": clean_text(abstract),
                }
            )
            if len(records) >= max_results:
                break
        next_cursor = payload.get("nextCursorMark")
        if not next_cursor or next_cursor == params["cursorMark"]:
            break
        params["cursorMark"] = next_cursor
    return records


def clean_text(text: str) -> str:
    """Strip HTML tags that Europe PMC embeds in abstracts and normalise spaces."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Step 2: NER
# --------------------------------------------------------------------------- #
def load_ner(model_name: str, entity_tag: str):
    """Load a token-classification pipeline.

    The BENT models use bare 'B'/'I' labels, which the transformers aggregation
    logic cannot group into spans, so we relabel them as B-<TAG>/I-<TAG>.
    They also tag every WordPiece sub-token as 'B' (e.g. rib/##ocic/##li/##b),
    so 'simple' aggregation fragments words; 'first' aggregates at word level
    (taking the first sub-token's label) and yields whole names like 'ribociclib'.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    id2label = {}
    for idx, label in model.config.id2label.items():
        id2label[int(idx)] = label if label == "O" or "-" in label else f"{label}-{entity_tag}"
    model.config.id2label = id2label
    model.config.label2id = {v: k for k, v in id2label.items()}
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first",
    )


def extract_entities(ner, sentences: list[str]) -> list[list[str]]:
    """Run NER over sentences; return cleaned entity strings per sentence."""
    outputs = ner(sentences, batch_size=16)
    per_sentence = []
    for sentence, ents in zip(sentences, outputs):
        names = []
        for ent in ents:
            if ent["score"] < MIN_ENTITY_SCORE:
                continue
            # Slice from the original text to keep correct casing/spacing.
            name = sentence[ent["start"] : ent["end"]].strip(" ,;:.()[]")
            if len(name) >= 2 and not name.startswith("##"):
                names.append(name)
        per_sentence.append(list(dict.fromkeys(names)))  # de-dupe, keep order
    return per_sentence


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-protein-ner",
        action="store_true",
        help="Skip the gene/protein NER model and detect HER2/ERBB2 by regex only.",
    )
    args = parser.parse_args()

    print(f"Querying Europe PMC: {SEARCH_QUERY!r}")
    records = fetch_abstracts(SEARCH_QUERY, MAX_RESULTS)
    print(f"  Retrieved {len(records)} abstracts.")
    if not records:
        print("No abstracts retrieved; aborting.")
        return 1

    print(f"Loading chemical NER model: {CHEMICAL_MODEL}")
    chem_ner = load_ner(CHEMICAL_MODEL, "CHEM")
    gene_ner = None
    if args.skip_protein_ner:
        print("Skipping gene/protein NER (HER2 detected by regex only).")
    else:
        print(f"Loading gene/protein NER model: {GENE_MODEL}")
        gene_ner = load_ner(GENE_MODEL, "GENE")

    rows = []
    kept_abstracts = 0
    for i, rec in enumerate(records, 1):
        sentences = split_sentences(rec["abstract"])
        chem_per_sent = extract_entities(chem_ner, sentences)
        gene_per_sent = (
            extract_entities(gene_ner, sentences)
            if gene_ner is not None
            else [[] for _ in sentences]
        )

        proteins = {g for genes in gene_per_sent for g in genes}
        mentions_her2 = any(HER2_PATTERN.search(p) for p in proteins) or bool(
            HER2_PATTERN.search(rec["abstract"])
        )
        has_chemical = any(chem_per_sent)
        print(
            f"  [{i:02d}/{len(records)}] PMID {rec['pmid']}: "
            f"{sum(map(len, chem_per_sent))} chem, {len(proteins)} protein mentions, "
            f"HER2={'yes' if mentions_her2 else 'no'}"
        )
        if not (mentions_her2 and has_chemical):
            continue

        kept_abstracts += 1
        for sentence, chems, genes in zip(sentences, chem_per_sent, gene_per_sent):
            for chem in chems:
                if HER2_PATTERN.fullmatch(chem):
                    continue  # occasionally tagged as a chemical
                rows.append(
                    {
                        "chemical_name": chem,
                        "context_sentence": sentence,
                        "pmid": rec["pmid"],
                        "her2_in_sentence": bool(HER2_PATTERN.search(sentence)),
                        "proteins_in_sentence": "; ".join(genes),
                        "title": rec["title"],
                    }
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "chemical_name",
            "context_sentence",
            "pmid",
            "her2_in_sentence",
            "proteins_in_sentence",
            "title",
        ],
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(
        f"\nKept {kept_abstracts}/{len(records)} abstracts; "
        f"wrote {len(df)} rows ({df['chemical_name'].nunique()} unique chemicals) "
        f"to {OUTPUT_CSV}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
