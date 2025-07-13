"""

Feature Selection Pipeline for Vital Status Prediction using Boruta-like Random Forest Filtering

This script performs feature selection on multi-modal gene expression data (protein-coding, lncRNA, miRNA)
by applying a Boruta-inspired wrapper method with Random Forests, followed by k-fold evaluation.

Outputs include:
- Final train/test splits for selected genes
- Feature selection frequency statistics
- Accuracy distributions
- Histograms and CSV summaries of selected features

Usage:
------
python feature_selection.py --data path_to_input.csv --ref path_to_reference.csv

Arguments:
--data: path to gene expression dataset (CSV with samples x genes + vital_status)
--ref:  path to reference gene list (must contain 'gene_id' and 'gene_type')

Assumptions:
------------
- Target variable is 'vital_status': 0 = dead, 1 = alive
- Gene types of interest: protein_coding, lncRNA, miRNA

Authors: James Brett and Bhargav Pulugundla 
"""

import pandas as pd
import os
import time
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Argument Parser --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Feature Selection using Boruta-style Random Forest")
    parser.add_argument('--data', type=str, required=True, help='Path to input expression dataset (CSV)')
    parser.add_argument('--ref', type=str, required=True, help='Path to gene reference file with gene types')
    return parser.parse_args()

# -------------------- Data Preparation --------------------
def split_data(df, test_size=0.2, random_state=42):
    """Split dataset while preserving class balance for vital_status."""
    df = df.sample(frac=1, random_state=random_state)
    X = df.drop(columns=['vital_status'])
    y = df['vital_status'].replace({'Dead': 0, 'Alive': 1})
    overall_balance = (y.sum() / len(y)) * 100

    while True:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        train_balance = (y_train.sum() / len(y_train)) * 100
        if abs(overall_balance - train_balance) < 3:
            break
        random_state += 1

    return X_train, X_test, y_train, y_test

# -------------------- Variance Filtering --------------------
def filter_var(df, mapper):
    """Filter genes based on variance threshold per gene type."""
    protein_cols = [col for col in df.columns if mapper.get(col) == 'protein_coding']
    mirna_cols   = [col for col in df.columns if mapper.get(col) == 'miRNA']
    lncRNA_cols  = [col for col in df.columns if mapper.get(col) == 'lncRNA']

    def keep_high_variance(sub_df, min_var):
        if sub_df.empty:
            return pd.DataFrame()
        var = sub_df.var(skipna=True)
        return sub_df[var[var >= min_var].index]

    protein_df = keep_high_variance(df[protein_cols], 0.1)
    mirna_df   = keep_high_variance(df[mirna_cols], 0.0)
    lncRNA_df  = keep_high_variance(df[lncRNA_cols], 0.01)

    return pd.concat([df_ for df_ in [protein_df, mirna_df, lncRNA_df] if not df_.empty], axis=1)

# -------------------- Boruta-like Selection --------------------
def Bortua(x_train, y_train, random_state, shadow_bias=1.2):
    """Run shadow feature-based importance selection (Boruta-like)."""
    shadow = x_train.apply(lambda col: col.sample(frac=1), axis=0)
    shadow.columns = [f'shadow_{col}' for col in x_train.columns]
    X_aug = pd.concat([x_train, shadow], axis=1)

    rf = RandomForestClassifier(
        n_estimators=1000, n_jobs=-1, max_features='sqrt',
        random_state=random_state, class_weight='balanced')
    rf.fit(X_aug, y_train)

    importances = pd.DataFrame({
        'feature': X_aug.columns,
        'importance': rf.feature_importances_
    })
    cutoff = importances[importances['feature'].str.startswith('shadow_')]['importance'].mean() * shadow_bias
    selected = importances[~importances['feature'].str.startswith('shadow_')]
    return selected[selected['importance'] > cutoff]['feature'].tolist()

# -------------------- K-Fold Wrapper --------------------
def k_fold_split(X, y, random_seed, k=5):
    """Repeated k-fold evaluation with Boruta-style selection."""
    num_runs = 200
    final_sets, accuracies = [], []

    while len(accuracies) < num_runs:
        random_seed += 1
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)

        for train_idx, val_idx in skf.split(X, y):
            x_tr, x_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            features = Bortua(x_tr, y_tr, random_seed)
            if not features:
                continue
            acc = accuracy_score(y_val, RandomForestClassifier().fit(x_tr[features], y_tr).predict(x_val[features]))
            if acc > 0.6:
                final_sets.append(features)
                accuracies.append(acc)

        print(f'✔ Successful runs: {len(accuracies)}/{num_runs}')
        if len(accuracies) == num_runs:
            break
    return final_sets, accuracies

# -------------------- Main Pipeline --------------------
def run_feature_selection(x_train, x_test, y_train, y_test, ref_df, n_feat, seed, name, mapper):
    """Execute full feature selection pipeline for a given gene type."""
    x_train = x_train[ref_df[ref_df['gene_type'] == name]['gene_id']]
    x_test = x_test[x_train.columns]
    x_train = filter_var(x_train, mapper)
    selected_sets, accs = k_fold_split(x_train, y_train, seed)

    feat_freq = defaultdict(int)
    for s in selected_sets:
        for feat in s:
            feat_freq[feat] += 1

    sorted_feats = sorted(feat_freq.items(), key=lambda x: x[1], reverse=True)
    pd.DataFrame(sorted_feats, columns=['feature', 'frequency']).to_csv(
        f'selected_genes/split_by_type/{name}_feature_selection_frequency.csv', index=False)

    plt.hist(list(feat_freq.values()), bins=50)
    plt.title(f"{name} Selection Histogram")
    plt.xlabel("Selection Count")
    plt.ylabel("Feature Count")
    plt.savefig(f'selected_genes/split_by_type/{name}feat_select_hist.png')
    plt.close()

    top_feats = [f for f, _ in sorted_feats][:n_feat]
    train_df = pd.concat([x_train[top_feats], y_train], axis=1)
    test_df = pd.concat([x_test[top_feats], y_test], axis=1)

    train_df.to_csv(f'selected_genes/split_by_type/{name}_train_df.csv')
    test_df.to_csv(f'selected_genes/split_by_type/{name}_test_df.csv')
    print(f"{name}: Mean accuracy: {np.mean(accs):.3f}")
    return train_df, test_df

# -------------------- Entry Point --------------------
if __name__ == '__main__':
    args = parse_args()
    df = pd.read_csv(args.data, index_col=0)
    ref_df = pd.read_csv(args.ref)[['gene_id', 'gene_type']]
    mapper = dict(zip(ref_df['gene_id'], ref_df['gene_type']))
    df.dropna(subset=['vital_status'], inplace=True)

    x_train, x_test, y_train, y_test = split_data(df)

    SEED = 1337
    run_feature_selection(x_train, x_test, y_train, y_test, ref_df, n_feat=1000, seed=SEED, name='protein_coding', mapper=mapper)
    run_feature_selection(x_train, x_test, y_train, y_test, ref_df, n_feat=500, seed=SEED, name='lncRNA', mapper=mapper)
    run_feature_selection(x_train, x_test, y_train, y_test, ref_df, n_feat=300, seed=SEED, name='miRNA', mapper=mapper)

