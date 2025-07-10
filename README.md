# AML-GNN-Survival

**AML-GNN-Survival** is a graph-based representation learning framework for survival prediction in Acute Myeloid Leukemia (AML). This repository implements a dual-graph GATv2 architecture that integrates transcriptomic profiles, clinical variables, and genome-scale metabolic models (GEMs) to model survival and prognosis.

This project accompanies the master’s thesis *“From Expression Profiles to Predictive Topologies: A Systems Approach to Prognostic Modeling in AML.”*

---

## Key Features

- **Dual-Graph Learning:** Jointly models gene–gene and patient–patient relationships using GATv2 layers.
- **Cross-Attention Module:** Bridges gene-level and patient-level embeddings for integrated prediction.
- **Multi-Objective:** Supports binary survival classification and continuous survival risk estimation.
- **Reproducible Pipeline:** Includes training scripts, graph construction, and configurable hyperparameters.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/bpulugundla/aml-gnn-survival.git
cd aml-gnn-survival
pip install -r requirements.txt
```
---

## How to run

```bash
python train.py --exp v1 --lr 1e-4 --batch_size 32 --epochs 100
```
---

## Project Structure

```
├── experiments # Various model runs
│   ├── Boruta-E4 # Saved experiment runs, logs, model outputs
│   ├── Boruta-GEMs-E6
│   └── GEMs-Weighted-E7
├── gems
├── LICENSE
├── README.md
├── requirements.txt # Python dependencies
└── utils # Common helper functions
    ├── map_patient_edges.py
    ├── regression.py
    └── utils.py
```
---

**Citation**
If you use this code, please cite:

> *Bhargav Pulugundla*, *From Expression Profiles to Predictive Topologies: A Systems Approach to Prognostic Modeling in AML*, Master’s Thesis, King's College London, 2025.

---

**License**
Released under the MIT License. See `LICENSE` for details.
