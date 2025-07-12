import torch
import numpy as np
import pandas as pd
import os


def extract_top_genes_attention(
    model, train_loader, edge_index_genes, gene_feature_cols, num_clinical, top_k=500
):
    """
    Extracts top genes based on average attention scores from the first gene-level GAT layer.
    Saves top_k genes and their scores to a CSV file.

    Args:
        model (nn.Module): Trained model with GAT layer.
        train_loader (DataLoader): DataLoader for training data.
        edge_index_genes (Tensor): Edge index for gene-gene graph.
        gene_feature_cols (list): List of gene feature names.
        num_clinical (int): Number of clinical feature columns.
        top_k (int): Number of top genes to extract
    """
    model.eval()
    gat_layer = model.gnn.gat_gene_1
    gene_attention_scores = {}

    with torch.no_grad():
        for batch_x, _, _, _ in train_loader:
            gene_x = batch_x[:, :-num_clinical].to(next(model.parameters()).device)
            x_input = gene_x.T  # [num_genes, batch_size]

            _, (edge_idx_used, attn_weights) = gat_layer(
                x_input, edge_index_genes, return_attention_weights=True
            )

            src_nodes = edge_idx_used[0].tolist()
            weights = attn_weights.squeeze().cpu().numpy()

            for src, weight in zip(src_nodes, weights):
                gene = gene_feature_cols[src]
                gene_attention_scores.setdefault(gene, []).append(weight)

    gene_attention_avg = {
        gene: float(np.mean(scores)) for gene, scores in gene_attention_scores.items()
    }

    sorted_genes = sorted(gene_attention_avg.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    df_top_genes = pd.DataFrame(sorted_genes, columns=["gene_id", "attention_score"])

    return df_top_genes


def get_top_genes_from_risk_gradients(csv_path, top_k=50, output_dir="analysis"):
    """
    Extracts and saves top-k genes associated with survival and death based on gradient scores.

    Args:
        csv_path (str): Path to the gene importance CSV.
        top_k (int): Number of top genes to return.
        output_dir (str): Directory to save the result CSVs.
    """
    df = pd.read_csv(csv_path)

    # Sort by signed gradient — positive = survival-associated
    top_alive_df = df.sort_values("risk_grad_mean_alive", ascending=False)[
        ["gene_id", "risk_grad_mean_alive"]
    ].head(top_k)

    # Sort by negative gradients — negative = death-associated
    top_dead_df = df.sort_values("risk_grad_mean_dead")[  # ascending=True by default
        ["gene_id", "risk_grad_mean_dead"]
    ].head(top_k)

    return top_alive_df, top_dead_df
