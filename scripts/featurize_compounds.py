"""Featurize mined compounds with Canonical SMILES and Morgan Fingerprints.

Workflow:
  1. Load data/mined_compounds.csv.
  2. Filter to keep only rows where her2_in_sentence is True.
  3. Convert chemical_name to lowercase, remove names containing numbers,
     and drop duplicates.
  4. Query the PubChem REST API to retrieve canonical SMILES for each compound.
  5. Convert SMILES to RDKit Mol objects and compute 1024-bit Morgan fingerprints
     (radius 2).
  6. Append SMILES and Fingerprint columns, drop compounds that failed the
     PubChem lookup or molecule parsing, and export to data/featurized_compounds.csv.
"""

from __future__ import annotations

import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

# --------------------------------------------------------------------------- #
# Paths & Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "mined_compounds.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "featurized_compounds.csv"

PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
)
PUBCHEM_TXT_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT"
)

FP_RADIUS = 2
FP_BITS = 1024


# --------------------------------------------------------------------------- #
# PubChem REST API
# --------------------------------------------------------------------------- #
def fetch_canonical_smiles(chemical_name: str, max_retries: int = 3) -> str | None:
    """Retrieve canonical SMILES string from PubChem PUG REST API for a chemical name."""
    encoded_name = urllib.parse.quote(chemical_name)
    url_json = PUBCHEM_PROPERTY_URL.format(name=encoded_name)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url_json, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                properties = data.get("PropertyTable", {}).get("Properties", [])
                if properties:
                    first_prop = properties[0]
                    # PubChem may return CanonicalSMILES, ConnectivitySMILES, or SMILES
                    smiles = (
                        first_prop.get("CanonicalSMILES")
                        or first_prop.get("ConnectivitySMILES")
                        or first_prop.get("SMILES")
                    )
                    if smiles:
                        return smiles.strip()
            elif resp.status_code == 404:
                # Compound name not found in PubChem
                return None
            elif resp.status_code in (503, 429):
                # Rate limit or temporary server busy; back off and retry
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                # Non-200 / non-404 status, try fallback TXT endpoint
                break
        except requests.RequestException:
            if attempt == max_retries - 1:
                break
            time.sleep(1.0 * (attempt + 1))

    # Fallback to plain text endpoint
    try:
        url_txt = PUBCHEM_TXT_URL.format(name=encoded_name)
        resp_txt = requests.get(url_txt, timeout=15)
        if resp_txt.status_code == 200 and resp_txt.text.strip():
            first_line = resp_txt.text.strip().splitlines()[0].strip()
            if first_line:
                return first_line
    except requests.RequestException:
        pass

    return None


# --------------------------------------------------------------------------- #
# RDKit Featurization
# --------------------------------------------------------------------------- #
def generate_morgan_fingerprint(
    smiles: str, radius: int = FP_RADIUS, n_bits: int = FP_BITS
) -> list[int] | None:
    """Generate a Morgan fingerprint bit vector as a Python list of integers."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        # Modern RDKit Morgan generator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    except Exception:
        # Backward compatibility fallback
        from rdkit.Chem import AllChem

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr.tolist()


# --------------------------------------------------------------------------- #
# Main Pipeline
# --------------------------------------------------------------------------- #
def main() -> int:
    if not INPUT_CSV.exists():
        print(f"Error: input file '{INPUT_CSV}' does not exist.")
        return 1

    print(f"1. Loading dataset from '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Loaded {len(df)} total rows.")

    # Step 1: Filter to rows where her2_in_sentence is True
    # Handle both boolean True and string "True"
    df = df[df["her2_in_sentence"].astype(str).str.strip().str.lower() == "true"].copy()
    print(f"2. Filtered to {len(df)} rows where 'her2_in_sentence' is True.")

    # Step 2: Convert chemical_name to lowercase & remove names containing numbers
    df["chemical_name"] = df["chemical_name"].astype(str).str.lower().str.strip()
    df = df[~df["chemical_name"].str.contains(r"\d", na=False)].copy()
    print(f"3. Kept {len(df)} rows after removing chemical names with numbers.")

    # Step 3: Drop duplicates
    df = df.drop_duplicates(subset=["chemical_name"]).reset_index(drop=True)
    print(f"4. Kept {len(df)} unique chemical names after dropping duplicates.")

    if df.empty:
        print("No chemicals remaining after filters. Exiting.")
        return 0

    # Step 4: Query PubChem for each unique chemical name
    print("\n5. Querying PubChem REST API for canonical SMILES...")
    smiles_cache: dict[str, str | None] = {}
    unique_names = df["chemical_name"].tolist()

    for idx, name in enumerate(unique_names, 1):
        if name not in smiles_cache:
            smiles = fetch_canonical_smiles(name)
            smiles_cache[name] = smiles
            status = "FOUND" if smiles else "NOT FOUND"
            print(f"   [{idx:02d}/{len(unique_names)}] '{name}' -> {status}")
            time.sleep(0.2)  # Respect PubChem usage policy (< 5 req/sec)

    df["smiles"] = df["chemical_name"].map(smiles_cache)

    # Step 5: Featurize with RDKit Morgan Fingerprints
    print(
        f"\n6. Generating {FP_BITS}-bit Morgan fingerprints (radius {FP_RADIUS}) with RDKit..."
    )
    fp_cache: dict[str, list[int] | None] = {}
    for name, smiles in smiles_cache.items():
        if smiles:
            fp_cache[name] = generate_morgan_fingerprint(smiles)
        else:
            fp_cache[name] = None

    df["fingerprint"] = df["chemical_name"].map(fp_cache)

    # Step 6: Drop rows that failed PubChem lookup or fingerprinting
    initial_count = len(df)
    df = df[df["smiles"].notna() & df["fingerprint"].notna()].copy()
    df = df.reset_index(drop=True)
    dropped_count = initial_count - len(df)
    print(
        f"   Kept {len(df)} compounds ({dropped_count} dropped due to missing SMILES/fingerprint)."
    )

    # Step 7: Save output dataset
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n7. Saved featurized dataset to '{OUTPUT_CSV}'.")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
