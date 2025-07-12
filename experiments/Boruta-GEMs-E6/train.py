import os
import numpy as np
import pandas as pd
import random
import logging
import argparse
import json
import glob
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    ConcatDataset,
    WeightedRandomSampler,
)
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import OneHotEncoder

from graphs import construct_gene_graph, construct_patient_graph
from model import GeneExpressionModel
from ...utils.map_patient_edges import remap_patient_edges
from ...utils.regression import fit_survival_regression

from extract_top_genes_attention import (
    extract_top_genes_attention,
    get_top_genes_from_risk_gradients,
)

from ...utils.utils import (
    set_seed,
    average_checkpoints,
    cleanup_all_checkpoints,
    EarlyStopping,
    SigmoidFocalLoss,
    plot_metric_curves,
)

# Set CUDA for Multi-GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    logging.info(f"Using {num_gpus} GPU(s)")
else:
    logging.info(f"No GPUs available. Using CPU(s)")

if torch.device("cpu"):
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(max(2, num_cores // 2))

SEED = 1337
set_seed(SEED)


def read_and_process_data(gene_exp, gene_map, clinical_data):
    """
    Reads and processes gene expression, gene mapping, and clinical data.
    Returns a merged DataFrame.
    """
    # Load datasets
    gene_expression = pd.read_csv(gene_exp)
    gene_mapping = pd.read_csv(gene_map)
    clinical_data = pd.read_csv(clinical_data)

    gene_expression.columns = [
        col.split(".")[0] if col.startswith("ENS") else col
        for col in gene_expression.columns
    ]

    gene_mapping = gene_mapping.drop(columns=["gene_name"])
    gene_mapping["gene_id"] = gene_mapping["gene_id"].str.split(".").str[0]

    clinical_subset = clinical_data[
        ["case_submitter_id", "ethnicity", "race", "age_at_diagnosis"]
    ].copy()

    # Remove duplicates and missing values in the subset
    clinical_subset = clinical_subset.drop_duplicates(
        subset=["case_submitter_id"]
    ).dropna(subset=["case_submitter_id"])

    clinical_subset = clinical_subset[
        clinical_subset["case_submitter_id"].str.contains(r"[a-zA-Z]", regex=True)
    ]
    clinical_subset = clinical_subset.rename(columns={"case_submitter_id": "case_id"})

    # Reshape gene expression data from wide to long format
    merged_data = gene_expression.melt(
        id_vars=["case_id", "vital_status"], var_name="gene_id", value_name="tpm"
    )

    # Merge with gene mapping (keeping gene_id and gene_type)
    merged_data = merged_data.merge(gene_mapping, on="gene_id", how="inner")

    # Convert case_id to string again after previous transformations
    merged_data["case_id"] = merged_data["case_id"].astype(str)

    merged_data = merged_data.merge(clinical_subset, on="case_id", how="left")

    merged_data = merged_data.dropna(
        subset=["vital_status", "ethnicity", "race", "age_at_diagnosis"]
    )

    merged_data["ethnicity"] = merged_data["ethnicity"].astype("category").cat.codes
    merged_data["race"] = merged_data["race"].astype("category").cat.codes
    merged_data["vital_status"] = merged_data["vital_status"].map(
        {"Alive": 1, "Dead": 0}
    )

    # Drop exact duplicates first
    merged_data = merged_data.drop_duplicates()

    # Remove 0.0 TPM rows only if there are duplicates for (case_id, gene_id)
    non_zero = merged_data[merged_data["tpm"] != 0.0]
    zero = merged_data[merged_data["tpm"] == 0.0]

    # Keep non-zero rows
    merged_data = pd.concat(
        [
            non_zero,
            zero[
                ~zero[["case_id", "gene_id"]]
                .apply(tuple, axis=1)
                .isin(non_zero[["case_id", "gene_id"]].apply(tuple, axis=1))
            ],
        ]
    )

    final_data = merged_data.pivot(index="case_id", columns="gene_id", values="tpm")

    final_data.reset_index(inplace=True)

    final_data = final_data.merge(
        merged_data[
            [
                "case_id",
                "ethnicity",
                "race",
                "age_at_diagnosis",
                "vital_status",
            ]
        ].drop_duplicates(),
        on="case_id",
        how="left",
    )

    return final_data


class GeneExpressionDataset(Dataset):
    def __init__(self, data, case_id_to_idx):
        self.case_ids = list(data["case_id"])
        self.case_id_to_idx = case_id_to_idx

        # Store days_to_death
        self.days_to_death = data["days_to_death"].fillna(0.0).astype("float32").values

        # Drop unused columns before tensor conversion
        data = data.drop(columns=["case_id", "days_to_death"])

        # Ensure all numerical columns are float32
        for col in data.columns:
            if col not in ["vital_status"]:
                data[col] = data[col].astype("float32")

        filtered_data = data.drop_duplicates()

        self.features = torch.tensor(
            filtered_data.drop(columns=["vital_status"]).values, dtype=torch.float32
        )

        self.labels = torch.tensor(
            filtered_data["vital_status"].values, dtype=torch.int64
        ).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        patient_idx = self.case_id_to_idx[case_id]
        days_to_death = torch.tensor(self.days_to_death[idx], dtype=torch.float32)
        return self.features[idx], self.labels[idx], patient_idx, days_to_death


def get_dataloaders(final_data, case_id_to_idx, batch_size=64):
    """
    Splits the data into train, validation, and test sets, ensuring all three gene_types are present in each split.
    Returns DataLoaders for PyTorch.
    """
    train_list, valid_list, test_list = [], [], []

    # Ensure all features are float32
    for col in final_data.columns:
        if col not in ["vital_status", "case_id"]:
            final_data[col] = final_data[col].astype("float32")

    final_data = final_data.drop_duplicates()

    dataset = GeneExpressionDataset(final_data, case_id_to_idx)

    # Define split sizes
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size  # Ensure all data is used

    generator = torch.Generator().manual_seed(SEED)
    # Perform stratified split
    train_subset, valid_subset, test_subset = random_split(
        dataset, [train_size, valid_size, test_size], generator=generator
    )

    # Append to lists
    train_list.append(train_subset)
    valid_list.append(valid_subset)
    test_list.append(test_subset)

    # final_data = final_data.drop(columns=["gene_type"], errors="ignore")

    # Concatenate datasets across all gene types
    datasets = {
        "train": ConcatDataset(train_list),
        "valid": ConcatDataset(valid_list),
        "test": ConcatDataset(test_list),
    }

    dataloaders = {}
    for phase in ["train", "valid", "test"]:
        dataset = datasets[phase]

        # Extract all labels
        targets = np.array([label for _, label, _, _ in dataset])

        # Compute class distribution
        class_counts = Counter(targets.flatten().tolist())
        total_samples = sum(class_counts.values())

        # Compute class weights (inverse frequency)
        class_weights = {
            cls: total_samples / count for cls, count in class_counts.items()
        }

        # print(f"{phase.capitalize()} Class Counts: {class_counts}")
        # print(f"{phase.capitalize()} Class Weights: {class_weights}")

        # Only apply weighted sampling for training
        sampler = None
        shuffle = False  # Shuffle only training data

        if phase == "train":
            sample_weights = np.array(
                [class_weights[int(label)] for label in targets.flatten()]
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=torch.Generator().manual_seed(SEED),
            )
        else:
            shuffle = True

        # Create DataLoader
        dataloaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            drop_last=True,
            sampler=sampler,
        )

    # Extract train, valid, and test loaders
    train_loader = dataloaders["train"]
    valid_loader = dataloaders["valid"]
    test_loader = dataloaders["test"]

    return train_loader, valid_loader, test_loader


def train_model(
    model,
    train_loader,
    valid_loader,
    edge_index_genes,
    edge_index_patients,
    num_clinical,
    checkpoint_dir,
    num_epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
):
    """
    Train the GATv2 Model with Feature Selection.

    Args:
    - model: The GATv2 model.
    - train_loader: DataLoader for training data.
    - valid_loader: DataLoader for validation data.
    - edge_index_genes: Gene graph edges.
    - edge_index_patients: Patient graph edges.
    - num_clinical: Number of clinical features
    - num_epochs: Number of training epochs.
    - lr: Learning rate.

    Returns:
    - Trained model
    """
    model = model.to(device)
    edge_index_genes = edge_index_genes.to(device)
    edge_index_patients = edge_index_patients.to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode="min", patience=5, factor=0.8
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    criterion = SigmoidFocalLoss(alpha=0.80, gamma=1)
    early_stopping = EarlyStopping(patience=20, delta=0.001)

    min_epochs = 30
    lambda_surv = 0.005
    best_val_auc = 0
    best_train_auc = 0
    epoch_auc_scores = {}
    train_auc_history, valid_auc_history = [], []
    train_metrics_by_class, valid_metrics_by_class = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_labels, train_preds = 0, [], []
        running_classification_loss = 0.0
        running_survival_loss = 0.0
        for batch_x, batch_y, batch_patient_ids, batch_days_to_death in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Separate clinical features from genes
            clinical_features = batch_x[:, -num_clinical:]
            gene_features = batch_x[:, :-num_clinical]

            filtered_edge_index_patients = remap_patient_edges(
                edge_index_patients, batch_patient_ids
            )

            optimizer.zero_grad()
            outputs, risk_scores = model(
                gene_features,
                edge_index_genes,
                filtered_edge_index_patients,
                clinical_features,
            )

            # logging.info(
            #    f"Logits: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}"
            # )

            # loss = compute_weighted_loss(outputs, batch_y.float())
            classification_loss = criterion(outputs, batch_y.float())

            # Survival loss for Dead cases
            flat_y = batch_y.view(-1)
            dead_mask = flat_y == 0

            if dead_mask.sum() > 0:
                survival_loss = F.mse_loss(
                    risk_scores[dead_mask], batch_days_to_death[dead_mask].log1p()
                )
            else:
                survival_loss = 0.0
            loss = classification_loss + lambda_surv * survival_loss

            running_classification_loss += classification_loss.item()
            running_survival_loss += survival_loss.item()

            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm**0.5
            # logging.info(f"Gradient Norm: {total_norm:.4f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Compute Training Metrics
        train_acc = accuracy_score(
            train_labels, (np.array(train_preds) > 0.5).astype(int)
        )
        train_auc = roc_auc_score(train_labels, train_preds)

        logging.info(
            f"Epoch {epoch+1}/{num_epochs} Classification Loss: {running_classification_loss/len(train_loader):.4f} "
            f"Survival Loss: {running_survival_loss/len(train_loader):.4f}"
        )
        train_preds_binary = (np.array(train_preds) > 0.5).astype(int)
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            train_labels, train_preds_binary, average=None, labels=[0, 1]
        )
        train_class_metrics = {
            "class_0_precision": precisions[0],
            "class_1_precision": precisions[1],
            "class_0_recall": recalls[0],
            "class_1_recall": recalls[1],
            "class_0_f1": f1s[0],
            "class_1_f1": f1s[1],
        }

        train_metrics_by_class.append(train_class_metrics)
        train_auc_history.append(train_auc)

        # Validate Model
        model.eval()
        valid_loss, valid_labels, valid_preds = 0, [], []

        valid_loss = 0
        valid_labels, valid_probs = [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_patient_ids, _ in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                clinical_features = batch_x[:, -num_clinical:]
                gene_features = batch_x[:, :-num_clinical]

                filtered_edge_index_patients = remap_patient_edges(
                    edge_index_patients, batch_patient_ids
                )

                outputs, _ = model(
                    gene_features,
                    edge_index_genes,
                    filtered_edge_index_patients,
                    clinical_features,
                )

                loss = criterion(outputs, batch_y.float()).item()
                valid_loss += loss

                valid_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                valid_labels.extend(batch_y.cpu().numpy())

        valid_loss /= len(valid_loader)

        # Compute precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            valid_labels, valid_probs
        )

        # Find threshold maximizing F1-score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        # Use this threshold for binarizing predictions
        valid_preds = (np.array(valid_probs) > optimal_threshold).astype(int)

        # Compute Validation Metrics
        valid_acc = accuracy_score(valid_labels, valid_preds)
        valid_auc = roc_auc_score(valid_labels, valid_probs)
        valid_precision = precision_score(valid_labels, valid_preds)
        valid_recall = recall_score(valid_labels, valid_preds)

        logging.info(
            f"Epoch {epoch+1}/{num_epochs}, Current LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        logging.info(
            f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}"
        )
        logging.info(
            f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid Precison: {valid_precision:.4f}, Valid Recall: {valid_recall:.4f}\n"
        )
        # save_checkpoint(model, optimizer, epoch, valid_acc, is_best=False)
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            valid_labels, valid_preds, average=None, labels=[0, 1]
        )
        valid_class_metrics = {
            "class_0_precision": precisions[0],
            "class_1_precision": precisions[1],
            "class_0_recall": recalls[0],
            "class_1_recall": recalls[1],
            "class_0_f1": f1s[0],
            "class_1_f1": f1s[1],
        }
        valid_metrics_by_class.append(valid_class_metrics)
        valid_auc_history.append(valid_auc)

        # Adjust Learning Rate
        # if epoch < warmup_epochs:
        #    warmup_scheduler.step()
        # else:
        # scheduler.step(loss)
        scheduler.step()

        torch.save({"model": model.state_dict()}, f"{checkpoint_dir}/epoch_{epoch}.pt")
        epoch_auc_scores[epoch] = valid_auc

        # Check Early Stopping
        if valid_auc > best_val_auc + 0.0001:
            best_val_auc = valid_auc
            patience_counter = 0
        elif epoch > min_epochs and train_auc < best_train_auc + 0.01:
            patience_counter += 1
        else:
            patience_counter = 0

        best_train_auc = max(best_train_auc, train_auc)

        if patience_counter >= 7:
            print("Early stopping triggered")
            break

    top_epochs = sorted(epoch_auc_scores, key=epoch_auc_scores.get, reverse=True)[:5]
    averaged_state_dict = average_checkpoints(
        top_epochs, os.path.join(checkpoint_dir, "epoch_{}.pt")
    )
    torch.save({"model": averaged_state_dict}, f"{checkpoint_dir}/model_top5_avg.pt")
    cleanup_all_checkpoints(checkpoint_dir=checkpoint_dir)

    return model, train_auc_history, valid_auc_history


def get_gems_genes(json_folder, min_model_fraction=0.10):
    gene_count = {}
    model_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    num_models = len(model_files)
    min_count = max(1, int(min_model_fraction * num_models))

    gene_file_name = f"gems_genes_num_models_{num_models}_min_count_{min_count}.txt"
    gene_file_path = gene_file_name

    if os.path.exists(gene_file_path):
        logging.info(f"Loading gems_genes from {gene_file_path}")
        with open(gene_file_path, "r") as f:
            gems_genes = {line.strip() for line in f}
    else:
        logging.info("Computing gems_genes from scratch...")
        for filename in model_files:
            with open(os.path.join(json_folder, filename), "r") as f:
                model_data = json.load(f)
                for gene in model_data.keys():
                    gene = gene.split(".")[0]  # Normalize gene ID
                    gene_count[gene] = gene_count.get(gene, 0) + 1

        gems_genes = {gene for gene, count in gene_count.items() if count >= min_count}

        with open(gene_file_path, "w") as f:
            f.write("\n".join(sorted(gems_genes)))

    return gems_genes


def filter_merged_data(
    merged_data, json_folder, key_genes_file, min_model_fraction=0.10
):
    """
    Reads gene IDs from model JSON files, keeps genes present in at least min_model_fraction of models,
    adds key genes, and filters merged_data accordingly.

    Args:
        merged_data (pd.DataFrame): The original merged dataset.
        json_folder (str): Path to folder containing model JSON files.
        key_genes_file (str): Path to key genes file.
        min_model_fraction (float): Minimum fraction of models a gene must appear in to be retained.

    Returns:
        pd.DataFrame: Filtered dataset containing only selected genes.
    """
    # Load key genes from file
    key_genes_df = pd.read_csv(key_genes_file)
    key_genes_df["gene_id"] = key_genes_df["gene_id"].str.split(".").str[0]
    key_genes = set(key_genes_df["gene_id"].tolist())

    gems_genes = get_gems_genes(json_folder)

    bortua_genes_file = "selected_gene_ids_Bortua.txt"

    with open(bortua_genes_file, "r") as f:
        bortua_genes = set([line.strip() for line in f if line.strip()])

    # Final set of selected genes
    selected_gene_ids = gems_genes | bortua_genes | key_genes

    # Filter merged_data columns
    gene_columns = [
        col.split(".")[0]
        for col in merged_data.columns
        if col.split(".")[0] in selected_gene_ids
    ]

    metadata_columns = [
        "case_id",
        "ethnicity",
        "race",
        "age_at_diagnosis",
        "vital_status",
    ]
    final_columns = metadata_columns + gene_columns

    merged_data_filtered = merged_data[final_columns]

    return merged_data_filtered


def get_gene_indices(merged_data_filtered):
    """
    Extracts indices of Boruta-selected genes from merged_data_filtered.

    Args:
        merged_data_filtered (pd.DataFrame): Filtered dataset containing selected genes.

    Returns:
        torch.Tensor: Tensor of selected gene indices.
    """
    # Extract ENS gene IDs from merged_data_filtered
    gene_columns = [
        col for col in merged_data_filtered.columns if col.startswith("ENS")
    ]

    # Create mapping of gene name to index
    gene_index_map = {gene: idx for idx, gene in enumerate(gene_columns)}

    # Get indices of selected genes
    selected_indices = torch.tensor(list(gene_index_map.values()), dtype=torch.long)

    return selected_indices


def preprocess_clinical_features(clinical_features):
    """
    Processes clinical features:
    - One-hot encodes ethnicity and race (categorical).
    - Log + Min-Max normalizes age_at_diagnosis (numerical).

    Returns:
    - Processed clinical features as a DataFrame, preserving original structure.
    """

    # Separate categorical & numerical columns
    ethnicity = clinical_features[["ethnicity"]].to_numpy()  # Categorical
    race = clinical_features[["race"]].to_numpy()  # Categorical
    age_at_diagnosis = clinical_features[["age_at_diagnosis"]].to_numpy()  # Numerical

    # One-hot encode ethnicity
    ethnicity_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ethnicity_encoded = ethnicity_encoder.fit_transform(ethnicity)

    # One-hot encode race
    race_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    race_encoded = race_encoder.fit_transform(race)

    # Create column names for ethnicity encoding
    ethnicity_categories = ethnicity_encoder.categories_[0]
    ethnicity_col_names = [
        f"ethnicity_{str(cat).replace(' ', '_')}" for cat in ethnicity_categories
    ]

    # Create column names for race encoding
    race_categories = race_encoder.categories_[0]
    race_col_names = [f"race_{str(cat).replace(' ', '_')}" for cat in race_categories]

    # Convert to DataFrame
    ethnicity_df = pd.DataFrame(
        ethnicity_encoded, columns=ethnicity_col_names, index=clinical_features.index
    )
    race_df = pd.DataFrame(
        race_encoded, columns=race_col_names, index=clinical_features.index
    )

    # Apply log transformation for age_at_diagnosis
    age_at_diagnosis = np.log1p(age_at_diagnosis)

    # Normalize using Min-Max Scaling
    scaler = MinMaxScaler()
    age_at_diagnosis = scaler.fit_transform(age_at_diagnosis)

    # Convert back to DataFrame
    age_df = pd.DataFrame(
        age_at_diagnosis, columns=["age_at_diagnosis"], index=clinical_features.index
    )

    # Merge processed features back into original structure
    processed_clinical = pd.concat([ethnicity_df, race_df, age_df], axis=1)

    return processed_clinical


def predict_on_test_set(
    model,
    test_loader,
    edge_index_genes,
    edge_index_patients,
    num_clinical,
    regressor=None,
    device="cuda",
):
    model.eval()
    model.to(device)

    all_probs = []
    all_preds = []
    all_labels = []
    all_case_ids = []
    all_risks = []
    estimated_days = []
    true_days_to_death = []

    with torch.no_grad():
        for batch_x, batch_y, batch_patient_ids, batch_days_to_death in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            clinical_features = batch_x[:, -num_clinical:]
            gene_features = batch_x[:, :-num_clinical]

            filtered_edge_index_patients = remap_patient_edges(
                edge_index_patients, batch_patient_ids
            )

            outputs, risk_scores = model(
                gene_features,
                edge_index_genes,
                filtered_edge_index_patients,
                clinical_features,
            )

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            risks = risk_scores.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy().flatten())
            all_risks.extend(risks)
            all_case_ids.extend(batch_patient_ids)

            true_days_to_death.extend(batch_days_to_death.cpu().numpy().tolist())

            # Predict days_to_death for predicted dead patients
            for pred, risk in zip(preds, risks):
                if pred == 0 and regressor is not None:
                    est_log_days = regressor.predict([[risk]])[0]
                    est_days = np.expm1(est_log_days)
                    estimated_days.append(est_days)
                else:
                    estimated_days.append(np.nan)

    results = pd.DataFrame(
        {
            "case_id": all_case_ids,
            "predicted_label": all_preds,
            "probability_dead": all_probs,
            "true_label": all_labels,
            "risk_score": all_risks,
            "true_days_to_death": true_days_to_death,
            "estimated_days_to_death": estimated_days,
        }
    )

    # Optionally print performance
    test_acc = accuracy_score(all_labels, all_preds)
    test_auc = roc_auc_score(all_labels, all_probs)
    logging.info(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

    dead_mask = (results["true_label"] == 0) & results["true_days_to_death"].notna()
    if regressor is not None and dead_mask.sum() > 0:
        errors = (
            results.loc[dead_mask, "estimated_days_to_death"]
            - results.loc[dead_mask, "true_days_to_death"]
        )
        mae = errors.abs().mean()
        rmse = (errors**2).mean() ** 0.5
        logging.info(f"Survival Prediction MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DualGraphGAT for AML survival prediction"
    )
    parser.add_argument(
        "--exp", type=str, default="v1", help="Experiment name for output directory"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    return parser.parse_args()


def main():

    args = parse_args()
    # Make output dir
    os.makedirs(args.exp, exist_ok=True)

    # Logging setup
    logging.basicConfig(
        filename=f"{args.exp}/training.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Starting experiment: {args.exp}")

    # Model Checkpoint Path
    checkpoint_dir = f"{args.exp}/checkpoints/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    batch_size = args.batch_size

    personalised_gems_jsons = "EMS/gene_gene_via_metabolite_noiso_stats"
    gene_exp_file = (
        "TCGA_blood_cancer_data/raw_gene_exp_vital_status/gene_exp_vital_status.csv"
    )
    gene_map_file = "TCGA_blood_cancer_data/selected_genes/split_by_type/gene_ref.csv"
    clinical_data_file = (
        "TCGA_blood_cancer_data/raw_gene_exp_vital_status/clinical_cleaned.csv"
    )
    key_genes_file = "Thesis/data/priv_gene_df.csv"

    merged_data_path = "cache/merged_data.pkl"
    if os.path.exists(merged_data_path):
        logging.info("Loading merged_data from cache ...")
        merged_data = pd.read_pickle(merged_data_path)
    else:
        logging.info("Reading data ...")
        merged_data = read_and_process_data(
            gene_exp_file, gene_map_file, clinical_data_file
        )
        os.makedirs(os.path.dirname(merged_data_path), exist_ok=True)
        merged_data.to_pickle(merged_data_path)

    alive_cases = merged_data[merged_data.vital_status == 1]
    dead_cases = merged_data[merged_data.vital_status == 0]

    alive_cases = alive_cases.sort_values("case_id")
    sampled_alive = alive_cases.sample(n=2 * len(dead_cases), random_state=SEED)
    merged_data_sampled = pd.concat([sampled_alive, dead_cases])

    merged_data_filtered = filter_merged_data(
        merged_data_sampled, personalised_gems_jsons, key_genes_file
    )

    logging.info("Constructing gene and patient graphs ...")
    # Construct Pearson correlation graph
    edge_index_genes, gene_ids = construct_gene_graph(
        merged_data_filtered.drop(
            columns=[
                "case_id",
                "vital_status",
                "ethnicity",
                "race",
                "age_at_diagnosis",
            ]
        ),
        top_k=50,
    )

    edge_index_patients, patient_ids = construct_patient_graph(
        merged_data_filtered.drop(
            columns=[
                "vital_status",
                "ethnicity",
                "race",
                "age_at_diagnosis",
            ]
        ),
        top_k=10,
    )

    case_id_to_idx = {case_id: idx for idx, case_id in enumerate(patient_ids)}

    # Filter merged_data to include only genes used in the graph
    merged_data_filtered = merged_data_filtered[
        [
            "case_id",
            "vital_status",
            "ethnicity",
            "race",
            "age_at_diagnosis",
        ]
        + list(gene_ids)
    ]  # Keep only retained genes

    merged_data_cleaned = merged_data_filtered.drop_duplicates(
        subset=[
            col
            for col in merged_data_filtered.columns
            if col not in ["case_id", "vital_status"]
        ]
    )

    processed_clinical = preprocess_clinical_features(
        merged_data_cleaned[["ethnicity", "race", "age_at_diagnosis"]]
    )

    # Drop original ethnicity column before merging
    merged_data_cleaned = merged_data_cleaned.drop(
        columns=["ethnicity"], errors="ignore"
    )
    merged_data_cleaned = merged_data_cleaned.drop(columns=["race"], errors="ignore")

    # Assign processed clinical features
    merged_data_cleaned = merged_data_cleaned.assign(**processed_clinical)

    # Identify clinical feature columns (ethnicity_*, race_* and age_at_diagnosis)
    clinical_cols = [
        col
        for col in merged_data_cleaned.columns
        if "ethnicity_" in col or "race_" in col
    ] + ["age_at_diagnosis"]

    # Move all clinical columns to the end
    feature_cols = [
        col for col in merged_data_cleaned.columns if col not in clinical_cols
    ]
    merged_data_cleaned = merged_data_cleaned[feature_cols + clinical_cols]

    num_ethnicities = merged_data_cleaned.filter(like="ethnicity_").shape[1]
    num_race = merged_data_cleaned.filter(like="race_").shape[1]

    gene_feature_cols = [
        gene for gene in gene_ids if gene in merged_data_cleaned.columns
    ]
    # Create mapping of gene name to index
    gene_index_map = {gene: idx for idx, gene in enumerate(gene_feature_cols)}
    gene_indices = torch.tensor(list(gene_index_map.values()), dtype=torch.long)

    # Adding days_to_death to predict survival (risk)
    clinical_data = pd.read_csv(clinical_data_file)
    clinical_days_to_death = clinical_data[
        ["case_submitter_id", "days_to_death"]
    ].copy()
    clinical_days_to_death = clinical_days_to_death.rename(
        columns={"case_submitter_id": "case_id"}
    )
    # Merge to merged_data_cleaned (your processed gene expression dataframe)
    merged_data_cleaned = merged_data_cleaned.merge(
        clinical_days_to_death, on="case_id", how="left"
    )

    logging.info("Creating data loaders ...\n")
    # Get DataLoaders (using filtered merged_data)
    train_loader, valid_loader, test_loader = get_dataloaders(
        merged_data_cleaned, case_id_to_idx, batch_size=batch_size
    )

    num_genes = len(gene_ids)
    clinical_dim = num_ethnicities + num_race + 1  # age_at_diagnosis
    hidden_dim = 768

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Total training cases: {len(train_loader.dataset)}")
    logging.info(f"Total validation cases: {len(valid_loader.dataset)}")
    logging.info(f"Total testing cases: {len(test_loader.dataset)}\n")

    model = GeneExpressionModel(
        batch_size, num_genes, clinical_dim, gene_indices, hidden_dim=hidden_dim
    ).to(device)

    logging.info("Training started ...\n")
    # Train the model
    trained_model, train_auc_history, valid_auc_history = train_model(
        model,
        train_loader,
        valid_loader,
        edge_index_genes,
        edge_index_patients,
        num_clinical=clinical_dim,
        checkpoint_dir=checkpoint_dir,
        num_epochs=100,
        lr=1e-4,
        weight_decay=5e-5,
    )

    plot_metric_curves(train_auc_history, valid_auc_history, output_dir=args.exp)

    avg_model = GeneExpressionModel(
        batch_size, num_genes, clinical_dim, gene_indices, hidden_dim=hidden_dim
    ).to(device)

    avg_model.load_state_dict(
        torch.load(f"{checkpoint_dir}/model_top5_avg.pt")["model"]
    )
    avg_model.eval()

    logging.info("Evaluating and predicting survival\n")
    regressor = fit_survival_regression(
        avg_model,
        train_loader,
        edge_index_genes,
        edge_index_patients,
        num_clinical=clinical_dim,
        device=device,
    )

    test_results_df = predict_on_test_set(
        model=avg_model,
        test_loader=test_loader,
        edge_index_genes=edge_index_genes,
        edge_index_patients=edge_index_patients,
        num_clinical=clinical_dim,
        regressor=regressor,
        device=device,
    )

    test_results_df.to_csv(
        f"{args.exp}/test_predictions_with_survival.csv", index=False
    )

    logging.info("Saving attention weights\n")
    top_k = 700
    df_top_genes = extract_top_genes_attention(
        avg_model,
        train_loader,
        edge_index_genes,
        gene_feature_cols,
        num_clinical=clinical_dim,
        top_k=top_k,
    )

    output_path = f"{args.exp}/top_{top_k}_genes_attention_updated.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_top_genes.to_csv(output_path, index=False)

    top_alive_df, top_dead_df = get_top_genes_from_risk_gradients(
        f"{args.exp}/gene_importance_risk.csv", top_k=top_k
    )

    top_alive_df.to_csv(f"{args.exp}/top_{top_k}_alive_genes.csv", index=False)
    top_dead_df.to_csv(f"{args.exp}/top_{top_k}_dead_genes.csv", index=False)

    logging.info("Training completed.")


if __name__ == "__main__":
    main()
