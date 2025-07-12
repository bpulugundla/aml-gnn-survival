from sklearn.linear_model import LinearRegression
import numpy as np

import torch
from utils import remap_patient_edges


def fit_survival_regression(
    model, train_loader, edge_index_genes, edge_index_patients, num_clinical, device
):
    """
    Fit a simple linear regression that maps risk scores from the GNN model
    to log-transformed survival times (days_to_death).

    This function:
    - Runs the model in evaluation mode on the training data.
    - Extracts risk scores for patients labeled as Dead (label == 0)
      who have non-missing days_to_death.
    - Applies a log1p transform to days_to_death to stabilize variance
      and reduce the impact of outliers.
    - Fits a scikit-learn LinearRegression mapping risk -> log1p(days_to_death).

    Args:
        model (torch.nn.Module): Trained graph model that outputs risk scores.
        train_loader (DataLoader): PyTorch DataLoader over the training set.
        edge_index_genes (torch.Tensor): Gene-gene edge index.
        edge_index_patients (torch.Tensor): Patient-patient edge index.
        num_clinical (int): Number of clinical features to split from batch_x.
        device (str): Device for inference.

    Returns:
        sklearn.linear_model.LinearRegression:
            Fitted linear regressor for predicting log-days-to-death from risk scores.

    Raises:
        ValueError: If no uncensored Dead patients are found in the training set.

    Notes:
        - The survival signal is only learned for Dead patients, since Alive patients
          are right-censored (unknown days_to_death).
        - Assumes risk scores are monotonically correlated with hazard — higher risk,
          lower expected survival. If this assumption fails, linear regression is inappropriate.
        - The log1p transform is inverted with expm1 during prediction.
    """
    model.eval()
    model.to(device)

    all_risks = []
    all_days = []

    with torch.no_grad():
        for batch_x, batch_y, batch_patient_ids, batch_days_to_death in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            clinical_features = batch_x[:, -num_clinical:]
            gene_features = batch_x[:, :-num_clinical]

            filtered_edge_index_patients = remap_patient_edges(
                edge_index_patients, batch_patient_ids
            )

            _, risk_scores = model(
                gene_features,
                edge_index_genes,
                filtered_edge_index_patients,
                clinical_features,
            )

            # Select only Dead patients with valid days_to_death
            mask = (batch_y.squeeze() == 0) & (batch_days_to_death > 0)
            if mask.any():
                selected_risks = risk_scores[mask].cpu().numpy()
                selected_days = batch_days_to_death[mask].cpu().numpy()
                all_risks.extend(selected_risks)
                all_days.extend(np.log1p(selected_days))  # log1p for stability

    if not all_risks:
        raise ValueError("No dead patients with valid days_to_death in training set.")

    regressor = LinearRegression()
    regressor.fit(np.array(all_risks).reshape(-1, 1), np.array(all_days))

    return regressor
