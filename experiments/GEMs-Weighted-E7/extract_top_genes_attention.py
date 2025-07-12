import torch
import numpy as np
import pandas as pd
import os

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from model import filter_edge_index


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
        for batch_x, _, _, _, _, _, _ in train_loader:
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


def get_top_genes_from_risk_gradients(csv_path, top_k=300, output_dir="analysis"):
    """
    Extracts and saves top-k genes associated with survival and death based on gradient scores.

    Args:
        csv_path (str): Path to the gene importance CSV.
        top_k (int): Number of top genes to return.
        output_dir (str): Directory to save the result CSVs.
    """
    df = pd.read_csv(csv_path)

    df["abs_alive"] = df["risk_grad_mean_alive"].abs()
    df["abs_dead"] = df["risk_grad_mean_dead"].abs()

    top_alive_df = (
        df.sort_values("abs_alive", ascending=False)[
            ["gene_id", "risk_grad_mean_alive"]
        ]
        .head(top_k)
        .reset_index(drop=True)
    )
    top_dead_df = (
        df.sort_values("abs_dead", ascending=False)[["gene_id", "risk_grad_mean_dead"]]
        .head(top_k)
        .reset_index(drop=True)
    )

    return top_alive_df, top_dead_df


def plot_attention_graph_gene(
    model,
    dataloader,
    edge_index_genes,
    num_nodes,
    num_clinical,
    node_index_to_name,
    topk=50,
    label="Graph",
):
    """
    Visualizes a gene-gene attention subgraph from the first GAT layer.

    This function runs the model in evaluation mode, collects edge attention scores
    from the first gene-level GAT layer, aggregates them across batches, computes
    per-node total attention, selects the top-k highest scoring nodes, filters edges
    connecting these nodes, and plots the resulting subgraph using NetworkX and Matplotlib.

    Args:
        model (torch.nn.Module): Trained model containing a gene-level GAT layer.
        dataloader (DataLoader): DataLoader providing input batches.
        edge_index_genes (torch.Tensor): Edge index for the gene-gene graph.
        num_nodes (int): Total number of gene nodes.
        num_clinical (int): Number of clinical features (to separate gene vs clinical inputs).
        node_index_to_name (dict): Mapping from node index to gene name.
        topk (int, optional): Number of top-ranking nodes to visualize. Default is 50.
        label (str, optional): Label for plot title and print statements. Default is "Graph".

    Saves:
        EPS file named "Gene_attention_graph_top{k}.eps" in the working directory.

    Example:
        plot_attention_graph_gene(model, dataloader, edge_index, 1000, 10, name_map)
    """
    edge_attn_store = defaultdict(list)
    gat_layer = model.gnn.gat_gene_1
    model.eval()
    with torch.no_grad():
        for batch_x, _, _, _, _, _, _ in dataloader:
            gene_x = batch_x[:, :-num_clinical].to(next(model.parameters()).device)
            x_input = gene_x.T  # [num_genes, batch_size]

            _, (edge_index_batch, alpha) = gat_layer(
                x_input, edge_index_genes, return_attention_weights=True
            )

            if alpha.dim() > 1:
                alpha = alpha.mean(dim=1)  # average heads

            for idx in range(edge_index_batch.shape[1]):
                i, j = edge_index_batch[:, idx].tolist()
                w = alpha[idx].item()
                # treat undirected: store (min, max)
                key = (min(i, j), max(i, j))
                edge_attn_store[key].append(w)

    print(f"{label}: Unique edges collected: {len(edge_attn_store)}")

    # Merge: mean
    final_edges = []
    final_weights = []
    for key, values in edge_attn_store.items():
        avg_w = sum(values) / len(values)
        final_edges.append(key)
        final_weights.append(avg_w)

    # Per-node attention sum
    node_score = torch.zeros(num_nodes)
    for (i, j), w in zip(final_edges, final_weights):
        node_score[i] += w
        node_score[j] += w

    # Top-k nodes
    _, top_nodes = torch.topk(node_score, topk)
    top_nodes_set = set(top_nodes.tolist())

    # Filter edges
    selected_edges = []
    selected_weights = []
    for (i, j), w in zip(final_edges, final_weights):
        if i in top_nodes_set and j in top_nodes_set:
            selected_edges.append((i, j))
            selected_weights.append(w)

    print(f"{label}: Edges in top-{topk} subgraph: {len(selected_edges)}")

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(top_nodes.tolist())
    for (i, j), w in zip(selected_edges, selected_weights):
        G.add_edge(i, j, weight=w)

    # Layout & plot
    plt.clf()
    pos = nx.spring_layout(G, seed=1337)
    fig, ax = plt.subplots(figsize=(12, 12))

    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="black", ax=ax)

    edges = G.edges(data=True)
    weights = [d["weight"] * 10 for (_, _, d) in edges]
    nx.draw_networkx_edges(
        G, pos, width=weights, edge_color=weights, edge_cmap=plt.cm.Blues, ax=ax
    )

    # Labels for nodes
    labels = {n: node_index_to_name[n] for n in G.nodes()}
    label_pos = {n: (x, y - 0.05) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels, font_size=10, ax=ax)

    # Colorbar for edges
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
    )
    sm._A = []
    fig.colorbar(sm, ax=ax, label="Mean Attention Weight (scaled)")

    ax.set_title(f"Top {topk} {label}s — Merged Attention Graph")
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig(f"Gene_attention_graph_top{topk}.eps", format="eps", dpi=600)


def export_attention_graph_gene_to_graphml(
    model,
    dataloader,
    edge_index_genes,
    num_nodes,
    num_clinical,
    node_index_to_name,
    topk=50,
    label="Gene",
    filename=None,
):
    """
    Extracts and exports a gene-gene attention subgraph from the first GAT layer to GraphML.

    This function runs the model in evaluation mode, collects edge attention scores
    from the first gene-level GAT layer, aggregates them across batches, computes
    per-node total attention, selects the top-k highest scoring nodes, filters edges
    connecting these nodes, and writes the resulting subgraph to a GraphML file.

    Args:
        model (torch.nn.Module): Trained model with a gene-level GAT layer.
        dataloader (DataLoader): DataLoader providing input batches.
        edge_index_genes (torch.Tensor): Edge index for the gene-gene graph.
        num_nodes (int): Total number of gene nodes.
        num_clinical (int): Number of clinical features (to separate gene vs clinical inputs).
        node_index_to_name (dict): Mapping from node index to gene name.
        topk (int, optional): Number of top-ranking nodes to keep. Default is 50.
        label (str, optional): Label for print statements. Default is "Gene".
        filename (str, optional): Output filename. If None, a default name is used.

    Returns:
        networkx.Graph: The exported subgraph with node and edge attributes.

    Example:
        G = export_attention_graph_gene_to_graphml(model, dataloader, edge_index, 1000, 10, name_map)
    """
    edge_attn_store = defaultdict(list)
    gat_layer = model.gnn.gat_gene_1

    model.eval()
    with torch.no_grad():
        for batch_x, _, _, _, _, _, _ in dataloader:
            gene_x = batch_x[:, :-num_clinical].to(next(model.parameters()).device)
            x_input = gene_x.T  # [num_genes, batch_size]

            _, (edge_index_batch, alpha) = gat_layer(
                x_input, edge_index_genes, return_attention_weights=True
            )

            if alpha.dim() > 1:
                alpha = alpha.mean(dim=1)  # average heads

            for idx in range(edge_index_batch.shape[1]):
                i, j = edge_index_batch[:, idx].tolist()
                w = alpha[idx].item()
                key = (min(i, j), max(i, j))  # treat as undirected
                edge_attn_store[key].append(w)

    print(f"{label}: Unique edges collected: {len(edge_attn_store)}")

    # Merge: mean
    final_edges = []
    final_weights = []
    for key, values in edge_attn_store.items():
        avg_w = sum(values) / len(values)
        final_edges.append(key)
        final_weights.append(avg_w)

    # Per-node total attention
    node_score = torch.zeros(num_nodes)
    for (i, j), w in zip(final_edges, final_weights):
        node_score[i] += w
        node_score[j] += w

    # Top-k nodes
    _, top_nodes = torch.topk(node_score, topk)
    top_nodes_set = set(top_nodes.tolist())

    # Filter edges
    selected_edges = []
    selected_weights = []
    for (i, j), w in zip(final_edges, final_weights):
        if i in top_nodes_set and j in top_nodes_set:
            selected_edges.append((i, j))
            selected_weights.append(w)

    print(f"{label}: Edges in top-{topk} subgraph: {len(selected_edges)}")

    # Build NetworkX graph with attributes
    G = nx.Graph()
    G.add_nodes_from(top_nodes.tolist())
    for (i, j), w in zip(selected_edges, selected_weights):
        G.add_edge(i, j, weight=w)

    for n in G.nodes():
        G.nodes[n]["name"] = node_index_to_name[n]
        G.nodes[n]["total_attention"] = node_score[n].item()

    # Write to GraphML
    if filename is None:
        filename = f"Gene_attention_graph_top{topk}.graphml"

    nx.write_graphml(G, filename)
    print(f"GraphML exported: {filename}")

    return G  # Optional: return NetworkX graph too if you want
