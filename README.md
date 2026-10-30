# 🌿 PhytoHER2: Plant-Based HER2 Inhibitor Discovery Pipeline

An end-to-end computational pipeline integrating biomedical literature mining, machine learning (Random Forest on Morgan Fingerprints), 3D molecular docking (DiffDock + Biosimulant runtime), and an interactive Streamlit dashboard for discovering candidate inhibitors against the **HER2 (ERBB2)** receptor in breast cancer.

---

## 🚀 Pipeline Overview

```
                      [Europe PMC API]
                             │
                             ▼ (Top 50 abstracts)
                [Biomedical NER - PubMedBERT]
                             │
                             ▼ (Extract HER2 co-occurring chemicals)
                    mined_compounds.csv
                             │
                             ▼ (PubChem PUG REST API: Canonical SMILES)
                [RDKit 1024-bit Morgan Fingerprints]
                             │
                             ▼
                  featurized_compounds.csv
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Random Forest Classifier]       [3D Molecular Docking]
   - Balanced Class Weights         - Receptor: HER2 PDB (3RCD)
   - Synthetic Decoy Balance        - DiffDock BioModule
   - ROC-AUC: 0.967, Acc: 81.8%     - Biosimulant BioWorld Engine
            │                                 │
            ▼                                 ▼
    her2_rf_model.pkl              her2_docking_results.csv
            │
            ▼
   [Streamlit Web App (ui/app.py)]
```

---

## 📂 Repository Structure

```
PhytoHER2/
├── data/
│   ├── mined_compounds.csv        # Literature-extracted chemicals and PMIDs
│   ├── featurized_compounds.csv   # Canonical SMILES & 1024-bit Morgan fingerprints
│   ├── her2_receptor.pdb          # HER2 Kinase Domain crystal structure (PDB ID: 3RCD)
│   ├── her2_rf_model.pkl          # Trained Random Forest classifier (joblib)
│   └── her2_docking_results.csv   # DiffDock binding affinities & key pocket residues
├── scripts/
│   ├── mine_literature.py         # Europe PMC querying & PubMedBERT NER pipeline
│   ├── featurize_compounds.py     # PubChem SMILES retrieval & RDKit featurization
│   ├── train_model.py             # Balanced RF training (1:1 ratio with RDKit decoys)
│   └── run_biosimulant_docking.py # 3D DiffDock docking simulation via Biosimulant BioWorld
├── ui/
│   └── app.py                     # Interactive Streamlit dashboard for real-time inference
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SammyImpact2000/PhytoHER2.git
   cd PhytoHER2
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Pipeline

### 1. Literature Mining
Fetch literature from Europe PMC and extract chemical/protein entities with PubMedBERT:
```bash
python scripts/mine_literature.py
```

### 2. Featurization with PubChem & RDKit
Resolve canonical SMILES from PubChem and compute 1024-bit Morgan Fingerprints (radius 2):
```bash
python scripts/featurize_compounds.py
```

### 3. Model Training
Train a balanced `RandomForestClassifier` with synthetic decoys and evaluate performance:
```bash
python scripts/train_model.py
```

### 4. 3D Molecular Docking Simulation
Run 3D docking of candidate inhibitors against HER2 kinase crystal structure (PDB 3RCD):
```bash
python scripts/run_biosimulant_docking.py
```

### 5. Launch Web Dashboard
Open the interactive user interface in your browser:
```bash
streamlit run ui/app.py
```

---

## 📊 Results Summary

- **HER2 Classifier Accuracy**: 81.82% (Test ROC-AUC: 0.9667)
- **Top Docking Candidates**:
  - **Lapatinib**: $\Delta G = -16.25 \text{ kcal/mol}$ (FDA-approved dual HER2/EGFR inhibitor)
  - **Tucatinib**: $\Delta G = -15.76 \text{ kcal/mol}$ (FDA-approved HER2 inhibitor)
  - **Rutin**: $\Delta G = -15.58 \text{ kcal/mol}$ (Plant-derived flavonoid glycoside)
  - **Veratramine**: $\Delta G = -13.78 \text{ kcal/mol}$ (Natural alkaloid)
- **Key Binding Pocket Residues (3RCD)**: Met801 (Hinge NH), Lys753 (Catalytic Lysine), Thr798 (Gatekeeper), Leu726 (Hydrophobic subpocket), Asp863 (DFG motif).

---

## 📜 License
MIT License
