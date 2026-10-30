"""Train a Random Forest classifier for HER2 inhibitor classification.

Pipeline:
  1. Load positive (mined) compounds and their 1024-bit Morgan fingerprints
     from data/featurized_compounds.csv.
  2. Generate 200 random decoy SMILES strings (using diverse rings and atoms)
     to serve as the negative class (Non-Inhibitor).
  3. Compute 1024-bit Morgan Fingerprints (radius 2) for the decoys.
  4. Combine datasets: label = 1 (Inhibitor) for mined compounds,
     label = 0 (Non-Inhibitor) for random decoys.
  5. Split dataset into 80% training and 20% testing sets (stratified).
  6. Train a scikit-learn RandomForestClassifier on the fingerprints.
  7. Evaluate and print Accuracy, F1-score, and ROC-AUC metrics to the terminal.
  8. Serialize the trained model to data/her2_rf_model.pkl using joblib.
"""

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Suppress RDKit parse/sanitization warnings for attempted random structures
RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# Paths & Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "featurized_compounds.csv"
OUTPUT_MODEL = PROJECT_ROOT / "data" / "her2_rf_model.pkl"

N_DECOYS = 200
FP_RADIUS = 2
FP_BITS = 1024
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Decoy Generation Blocks (Diverse Rings, Atoms, and Linkers)
# --------------------------------------------------------------------------- #
SUBSTITUENTS = [
    "C", "CC", "CCC", "C(C)C", "C(C)(C)C",
    "O", "CO", "CCO", "N", "CN", "CCN", "N(C)C", "N(CC)CC",
    "C(=O)O", "C(=O)N", "C(=O)C", "C(=O)CC",
    "F", "Cl", "Br", "C#N",
    "S(=O)(=O)C", "S(=O)(=O)N", "C(F)(F)F", "OC(F)(F)F",
    "c1ccccc1", "c1ccncc1", "c1ccsc1", "c1ccoc1",
    "C1CCCCC1", "C1CCCC1", "C1CCOCC1", "C1CCNCC1", "C1CNCCN1",
    "c1ccc2ccccc2c1",
]

LINKERS = [
    "", "C", "CC", "CCC", "O", "CO", "COC", "S",
    "C(=O)", "C(=O)N", "NC(=O)", "C(=O)O", "OC(=O)", "S(=O)(=O)",
]

RING_TEMPLATES = [
    # Monosubstituted ring cores
    "c1cc({X})ccc1", "c1cc({X})cnc1", "c1cc({X})c(O)cc1", "c1cc({X})c(N)cc1",
    "c1cc({X})c(F)cc1", "c1cc({X})c(Cl)cc1", "c1cc({X})c(C)cc1",
    "C1CC({X})CCC1", "C1CC({X})CNC1", "C1CCN({X})CC1", "C1COCC({X})C1",
    "C1CC({X})CC1", "c1cc({X})sc1", "c1cc({X})oc1", "c1cc({X})c[nH]1",
    # Disubstituted ring cores
    "c1cc({X})ccc1{Y}", "c1cc({X})cc({Y})c1", "c1c({X})cc({Y})cc1",
    "c1cc({X})cnc1{Y}", "C1CC({X})CCC1{Y}", "C1CCN({X})CC1{Y}",
]


def generate_decoy_smiles(
    n_decoys: int = N_DECOYS, seed: int = RANDOM_STATE
) -> list[str]:
    """Generate `n_decoys` unique, chemically valid decoy SMILES with diverse rings and atoms."""
    rng = random.Random(seed)
    decoys: set[str] = set()
    attempts = 0
    max_attempts = n_decoys * 20

    while len(decoys) < n_decoys and attempts < max_attempts:
        attempts += 1
        template = rng.choice(RING_TEMPLATES)
        sub1 = rng.choice(SUBSTITUENTS)
        lnk1 = rng.choice(LINKERS)
        x_group = f"{lnk1}{sub1}" if lnk1 else sub1

        if "{Y}" in template:
            sub2 = rng.choice(SUBSTITUENTS)
            lnk2 = rng.choice(LINKERS)
            y_group = f"{lnk2}{sub2}" if lnk2 else sub2
            cand_smi = template.format(X=x_group, Y=y_group)
        else:
            cand_smi = template.format(X=x_group)

        mol = Chem.MolFromSmiles(cand_smi)
        if mol is not None:
            can_smi = Chem.MolToSmiles(mol)
            decoys.add(can_smi)

    if len(decoys) < n_decoys:
        raise RuntimeError(
            f"Could only generate {len(decoys)} decoys (requested {n_decoys})."
        )

    return sorted(decoys)[:n_decoys]


def compute_morgan_fingerprint(
    smiles: str, radius: int = FP_RADIUS, n_bits: int = FP_BITS
) -> np.ndarray:
    """Compute 1024-bit Morgan fingerprint array for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr


# --------------------------------------------------------------------------- #
# Main Workflow
# --------------------------------------------------------------------------- #
def main() -> int:
    # Step 1: Load mined positive compounds
    if not INPUT_CSV.exists():
        print(f"Error: input file '{INPUT_CSV}' does not exist.")
        return 1

    print(f"1. Loading positive dataset from '{INPUT_CSV}'...")
    pos_df = pd.read_csv(INPUT_CSV)
    print(f"   Loaded {len(pos_df)} mined inhibitor compounds (Positives).")

    # Parse stored fingerprint lists/arrays
    X_pos = np.array(
        [
            ast.literal_eval(fp) if isinstance(fp, str) else list(fp)
            for fp in pos_df["fingerprint"]
        ],
        dtype=int,
    )
    y_pos = np.ones(len(X_pos), dtype=int)

    # Step 2: Generate random decoys (1:1 ratio with true inhibitors)
    n_decoys = len(X_pos)
    print(f"\n2. Generating {n_decoys} random decoy SMILES strings using RDKit (1:1 balanced ratio)...")
    decoy_smiles = generate_decoy_smiles(n_decoys, seed=RANDOM_STATE)
    print(f"   Successfully generated {len(decoy_smiles)} unique valid decoy molecules.")

    # Step 3: Compute Morgan fingerprints for decoys
    print(f"3. Computing {FP_BITS}-bit Morgan Fingerprints (radius {FP_RADIUS}) for decoys...")
    X_neg = np.array(
        [compute_morgan_fingerprint(smi) for smi in decoy_smiles],
        dtype=int,
    )
    y_neg = np.zeros(len(X_neg), dtype=int)

    # Step 4: Combine datasets
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    print(f"\n4. Combined Dataset:")
    print(f"   Total samples: {len(y)}")
    print(f"   - Class 1 (Inhibitor / Mined): {np.sum(y == 1)}")
    print(f"   - Class 0 (Non-Inhibitor / Decoy): {np.sum(y == 0)}")
    print(f"   Feature matrix shape: {X.shape}")

    # Step 5: 80/20 Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n5. Train/Test Split (80/20, stratified):")
    print(
        f"   - Training set: {len(y_train)} samples "
        f"({np.sum(y_train == 1)} pos, {np.sum(y_train == 0)} neg)"
    )
    print(
        f"   - Testing set:  {len(y_test)} samples "
        f"({np.sum(y_test == 1)} pos, {np.sum(y_test == 0)} neg)"
    )

    # Step 6: Train Random Forest Classifier
    print(f"\n6. Training RandomForestClassifier on 1024-bit Morgan Fingerprints...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    # Step 7: Evaluate Model
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION METRICS (Test Set)")
    print("=" * 50)
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("=" * 50)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   [[TN={cm[0, 0]:2d}, FP={cm[0, 1]:2d}],")
    print(f"    [FN={cm[1, 0]:2d}, TP={cm[1, 1]:2d}]]")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Non-Inhibitor (0)", "Inhibitor (1)"],
            digits=4,
        )
    )

    # Step 8: Save trained model using joblib
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, OUTPUT_MODEL)
    print(f"7. Successfully saved trained model to '{OUTPUT_MODEL}'.")
    print(f"   Model file size: {OUTPUT_MODEL.stat().st_size / 1024:.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
