import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import dense_to_sparse

import logging


def construct_gene_graph(data, top_k=50, key_genes=None):
    """
    Constructs a gene-gene interaction graph using Pearson correlation.
    Ensures priority genes are included even if they don't meet the top-k threshold.

    - data: DataFrame where rows are samples and columns are gene expression values.
    - top_k: Number of strongest edges to keep per gene.
    - key_genes: List of gene names to always include.

    Returns:
    - edge_index (torch.Tensor)
    - gene_ids (list)
    """
    data = data.drop_duplicates()
    gene_cols = [col for col in data.columns if col.startswith("ENS")]
    gene_data = data[gene_cols]

    num_genes = len(gene_cols)

    # Remove genes with zero variance
    gene_data = gene_data.loc[:, gene_data.std() > 1e-6]

    gene_ids = gene_data.columns.tolist()
    num_genes = len(gene_ids)

    # Compute Pearson correlation matrix (vectorized)
    gene_matrix = gene_data.to_numpy()
    correlation_matrix = np.corrcoef(gene_matrix, rowvar=False)

    # Take absolute values for ranking correlations
    abs_corr_matrix = np.abs(correlation_matrix)

    # Get top-k correlated indices for all genes at once
    top_k_indices = np.argpartition(-abs_corr_matrix, top_k + 1, axis=1)[:, : top_k + 1]

    if key_genes:
        # Map priority genes to indices
        key_gene_indices = {
            gene_ids.index(gene) for gene in key_genes if gene in gene_ids
        }

        # Ensure key genes are always included
        for key_gene in key_gene_indices:
            key_neighbors = np.argpartition(-abs_corr_matrix[key_gene], top_k + 1)[
                : top_k + 1
            ]
            key_neighbors = set(key_neighbors).union(
                set(top_k_indices[key_gene])
            )  # Merge sets to avoid duplicates
            top_k_indices[key_gene] = list(key_neighbors)[
                : top_k + 1
            ]  # Ensure it doesn't exceed top_k + 1

    # Filter out self-loops efficiently
    row_indices, col_indices = np.where(np.isin(top_k_indices, np.arange(num_genes)))
    edges = set(zip(row_indices, top_k_indices[row_indices, col_indices]))

    logging.info(
        f"Keeping up to {top_k} strongest connections for each of the {num_genes} genes"
    )

    edge_index = torch.tensor(np.array(list(edges)).T, dtype=torch.long)

    return edge_index, gene_ids


def construct_patient_graph(data, top_k=10):
    """
    Constructs a patient-patient similarity graph using both gene expression.

    - data: DataFrame where rows are patients and columns are gene expression values.
    - top_k: Number of strongest patient connections to retain.

    Returns:
    - edge_index (torch.Tensor): Sparse adjacency matrix in PyG format.
    """
    patient_ids = set(list(data["case_id"]))

    data = data.drop(
        columns=[
            "case_id",
        ]
    )
    data = data.drop_duplicates()

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(data)

    # Convert to distance matrix (1 - cosine similarity)
    distance_matrix = 1 - similarity_matrix

    # Keep only top-k similar patients per row
    adjacency_matrix = np.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        top_k_indices = np.argsort(distance_matrix[i])[: top_k + 1]  # Include self-loop
        adjacency_matrix[i, top_k_indices] = similarity_matrix[i, top_k_indices]

    logging.info(
        f"Keeping up to {top_k} strongest connections for each of the {len(patient_ids)} patients\n"
    )

    # Convert to edge_index format for PyG
    edge_index = dense_to_sparse(torch.tensor(adjacency_matrix, dtype=torch.float32))[0]

    return edge_index, patient_ids
