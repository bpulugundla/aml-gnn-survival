import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class DualGraphGAT(nn.Module):
    def __init__(
        self,
        batch_size,
        num_genes,
        num_clinical,
        hidden_dim=512,
        num_classes=1,
        dropout=0.2,
    ):
        """
        DualGraphGAT implements a multi-layer Graph Attention Network for
        gene-level and patient-level graph learning in AML prognosis tasks.

        Args:
            batch_size (int): Input feature dimension for GAT layers.
            num_genes (int): Number of gene features.
            num_clinical (int): Number of clinical features.
            hidden_dim (int): Hidden dimension for patient-level GAT.
            num_classes (int): Output dimension (default=1 for binary).
            dropout (float): Dropout probability.
        """
        super(DualGraphGAT, self).__init__()

        # self.temperature = nn.Parameter(torch.ones(1) * temperature)
        # Learnable centrality weights
        self.alpha = nn.Parameter(torch.tensor(0.4))
        self.beta = nn.Parameter(torch.tensor(0.4))
        self.gamma = nn.Parameter(torch.tensor(0.2))

        # Gene-Gene GAT layers
        self.gat_gene_1 = GATv2Conv(
            batch_size, batch_size, heads=2, concat=True, dropout=dropout
        )
        self.bn_gene_1 = nn.LayerNorm(batch_size * 2)
        self.gat_gene_2 = GATv2Conv(
            batch_size * 2, batch_size, heads=2, concat=False, dropout=dropout
        )
        self.bn_gene_2 = nn.LayerNorm(batch_size)

        # Gene-Gene GAT Layer 3
        self.gat_gene_3 = GATv2Conv(
            batch_size, batch_size, heads=2, concat=False, dropout=dropout
        )
        self.bn_gene_3 = nn.LayerNorm(batch_size)

        # Patient-Patient GAT layers
        self.gat_patient_1 = GATv2Conv(
            hidden_dim, hidden_dim, heads=2, concat=True, dropout=dropout
        )
        self.bn_patient_1 = nn.LayerNorm(hidden_dim * 2)
        self.gat_patient_2 = GATv2Conv(
            hidden_dim * 2, hidden_dim, heads=2, concat=False, dropout=dropout
        )
        self.bn_patient_2 = nn.LayerNorm(hidden_dim)

        # Patient-Patient GAT Layer 3
        self.gat_patient_3 = GATv2Conv(
            hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout
        )
        self.bn_patient_3 = nn.LayerNorm(hidden_dim)

        # Feature projection layers
        self.fc_gene = nn.Linear(num_genes, hidden_dim)
        self.fc_patient = nn.Linear(hidden_dim, hidden_dim)
        self.fc_gene_to_patient = nn.Linear(num_genes, hidden_dim)

        # Define cross-attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

        self.clinical_encoder = nn.Linear(num_clinical, num_clinical)

        self.risk_head = nn.Sequential(
            nn.Linear((hidden_dim * 2) + num_clinical, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Classification layers
        self.fc1 = nn.Linear((hidden_dim * 2) + num_clinical, hidden_dim // 2)
        self.bn_fc = nn.LayerNorm(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 32)
        self.bn_fc_2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights for the model, including GAT layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, GATv2Conv):
                # Correct initialization for GATv2Conv
                if hasattr(m, "lin_r") and isinstance(m.lin_r, nn.Linear):
                    nn.init.xavier_uniform_(m.lin_r.weight)
                    if m.lin_r.bias is not None:
                        nn.init.zeros_(m.lin_r.bias)

    def forward(
        self,
        x_genes,
        edge_index_genes,
        edge_index_patients,
        clinical_features,
        closeness_scores,
        eigenvector_scores,
        betweenness_scores,
    ):
        """
        Forward pass of the DualGraphGAT model with learnable centrality weighting
        and hierarchical attention over gene and patient graphs.

        This method:
        1. Computes a learnable centrality-weighted scaling of the raw gene expression.
        2. Passes the scaled gene features through stacked gene-gene GAT layers,
           with residual connections to preserve low-level signal.
        3. Transforms the gene-level embeddings into patient-level representations.
        4. Passes the patient embeddings through stacked patient-patient GAT layers,
           again with residual connections.
        5. Computes cross-attention between final gene and patient embeddings.
        6. Concatenates the combined embeddings with explicit clinical features.
        7. Produces both:
           - A classification output (e.g., alive/dead probability logits).
           - A scalar risk score for downstream survival regression.

        Args:
            x_genes (torch.Tensor): Raw gene expression matrix of shape [batch_size, num_genes].
            edge_index_genes (torch.Tensor): Edge index tensor for the gene-gene graph.
            edge_index_patients (torch.Tensor): Edge index tensor for the patient-patient graph (filtered for batch).
            clinical_features (torch.Tensor): Clinical covariates, shape [batch_size, num_clinical].
            closeness_scores (torch.Tensor): Node-level closeness centrality scores, shape [batch_size, num_genes].
            eigenvector_scores (torch.Tensor): Node-level eigenvector centrality scores, shape [batch_size, num_genes].
            betweenness_scores (torch.Tensor): Node-level betweenness centrality scores, shape [batch_size, num_genes].

        Returns:
            tuple:
                - x_outputs (torch.Tensor): Final logits for classification, shape [batch_size, num_classes].
                - risk_score (torch.Tensor): Scalar risk score per patient, shape [batch_size].

        Notes:
            - The centrality weighting is learnable via coefficients alpha, beta, and gamma.
            - Gene-level GAT layers operate over the transposed feature matrix to align
              with the gene graph structure.
            - Residual connections help retain signal and stabilize deeper GAT stacks.
            - Cross-attention enables the model to learn interactions between gene-level
              and patient-level features before final classification and risk estimation.
            - The output `risk_score` can be passed to a survival regressor or used directly.
        """

        # Compute combined centrality weights (shape: [batch_size, num_genes])
        centrality_score = (
            self.alpha * closeness_scores
            + self.beta * eigenvector_scores
            + self.gamma * betweenness_scores
        )

        # Scale gene features by centrality weights
        x_genes = x_genes * centrality_score

        # Process gene-gene relationships
        x_genes_gat = self.gat_gene_1(
            x_genes.T, edge_index_genes
        )  # Transpose for gene relationships
        x_genes_gat = self.bn_gene_1(x_genes_gat)
        x_genes_gat = F.leaky_relu(x_genes_gat)
        x_genes_gat = self.gat_gene_2(x_genes_gat, edge_index_genes)
        x_genes_gat = self.bn_gene_2(x_genes_gat)
        x_genes_gat = F.leaky_relu(x_genes_gat)

        # Residual Connection
        res_genes = x_genes_gat

        # Gene GAT Layer 3
        x_genes_gat = self.gat_gene_3(x_genes_gat, edge_index_genes)
        x_genes_gat = self.bn_gene_3(x_genes_gat)
        x_genes_gat = F.leaky_relu(x_genes_gat)

        # Add Residual
        x_genes_gat = x_genes_gat + res_genes
        x_genes_gat = x_genes_gat.T
        x_genes_embed = self.fc_gene(x_genes_gat)

        # Transform gene expression into meaningful patient-level features
        x_patients_embed = self.fc_gene_to_patient(x_genes)

        # Apply Patient-Patient GAT
        x_patients_gat = self.gat_patient_1(x_patients_embed, edge_index_patients)
        x_patients_gat = self.bn_patient_1(x_patients_gat)
        x_patients_gat = F.leaky_relu(x_patients_gat)
        x_patients_gat = self.gat_patient_2(x_patients_gat, edge_index_patients)
        x_patients_gat = self.bn_patient_2(x_patients_gat)
        x_patients_gat = F.leaky_relu(x_patients_gat)

        # Residual Connection
        res_patients = x_patients_gat

        # Patient GAT Layer 3
        x_patients_gat = self.gat_patient_3(x_patients_gat, edge_index_patients)
        x_patients_gat = self.bn_patient_3(x_patients_gat)
        x_patients_gat = F.leaky_relu(x_patients_gat)

        # Add Residual
        x_patients_gat = x_patients_gat + res_patients

        x_patients_embed = self.fc_patient(x_patients_gat)

        # Compute cross-attention between gene embeddings and patient embeddings
        x_cross_attn, _ = self.cross_attn(
            x_genes_embed.unsqueeze(0),  # Queries: genes
            x_patients_embed.unsqueeze(0),  # Keys: patients
            x_patients_embed.unsqueeze(0),  # Values: patients
        )
        x_cross_attn = x_cross_attn.squeeze(0)  # Remove extra dim

        # Concatenate the cross-attended embeddings with clinical features
        clinical_features_embed = self.clinical_encoder(clinical_features)
        x_combined = torch.cat(
            [x_genes_embed, x_patients_embed, clinical_features_embed], dim=1
        )
        risk_score = self.risk_head(x_combined).squeeze(-1)

        # Final Classification
        x_outputs = F.relu(self.bn_fc(self.fc1(x_combined)))
        x_outputs = self.dropout(x_outputs)
        x_outputs = F.relu(self.bn_fc_2(self.fc2(x_outputs)))
        x_outputs = self.dropout(x_outputs)
        x_outputs = self.fc3(x_outputs)

        return x_outputs, risk_score


def filter_edge_index(edge_index, top_k_indices):
    """
    Filters and remaps edge_index to match the top-k selected genes.

    - edge_index: Full gene-gene edge index (torch.Tensor of shape [2, num_edges])
    - top_k_indices: Selected top-k gene indices (torch.Tensor of shape [500])

    Returns:
    - filtered_edge_index: Edge index with only selected gene connections, remapped to [0, 499].
    """
    top_k_list = top_k_indices.tolist()
    top_k_set = set(top_k_list)  # Convert to set for fast lookup

    # Create mapping {original_gene_id -> new_index}
    gene_id_map = {old_id: new_idx for new_idx, old_id in enumerate(top_k_list)}

    # Keep edges where both nodes exist in the top-k gene set
    mask = [
        (gene_id_map[src], gene_id_map[dst])  # Remap gene IDs
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist())
        if src in top_k_set and dst in top_k_set
    ]

    if not mask:  # No valid edges remaining
        return torch.empty((2, 0), dtype=torch.long)

    # Convert to tensor
    filtered_edge_index = torch.tensor(mask, dtype=torch.long).T

    return filtered_edge_index


class GeneExpressionModel(nn.Module):
    def __init__(
        self,
        batch_size,
        num_genes,
        num_clinical,
        gene_indices,
        hidden_dim=512,
        num_classes=1,
    ):
        """
        GeneExpressionModel combines a dual-graph attention mechanism
        for learning from both gene-gene and patient-patient relationships,
        augmented with explicit clinical features.

        Architecture:
        - Takes raw gene expression vectors for each patient.
        - Filters the gene-gene edge index to retain only selected top-K genes.
        - Passes gene features and clinical features through a DualGraphGAT,
          which attends across both graphs.
        - Outputs logits (for binary classification) and optionally risk scores
          for downstream survival regression.

        Args:
            batch_size (int): Number of samples per batch.
            num_genes (int): Number of gene features.
            num_clinical (int): Number of clinical features (e.g., one-hot encoded ethnicity/race + age).
            gene_indices (torch.Tensor): Indices of selected genes to filter the gene graph.
            hidden_dim (int, optional): Hidden dimensionality for GNN layers. Default: 512.
            num_classes (int, optional): Output dimensionality (1 for binary classification).

        Notes:
            - The model assumes that `filter_edge_index` correctly prunes the gene graph
              to match the selected genes in `gene_indices`. Any mismatch between input
              `x` columns and graph edges will yield incorrect message passing.
            - The patient graph must be pre-filtered to match the current batch's patients.
        """
        super(GeneExpressionModel, self).__init__()

        self.gene_indices = gene_indices
        self.gnn = DualGraphGAT(
            batch_size, num_genes, num_clinical, hidden_dim, num_classes, dropout=0.2
        )

    def forward(
        self,
        x,
        edge_index_genes,
        filtered_edge_index_patients,
        clinical_features,
        closeness_scores,
        eigenvector_scores,
        betweenness_scores,
    ):
        """
        x: Raw gene expression data [batch_size, num_genes]
        edge_index_genes: Gene-gene graph
        filtered_edge_index_patients: Patient-patient graph filtered to contain the patients edges in the batch
        clinical_features: [batch_size, 2] (ethnicity, race, age_at_diagnosis)
        """
        # Filtering gene edges to contain top k genes only
        filtered_edge_index_genes = filter_edge_index(
            edge_index_genes, self.gene_indices
        )

        # Process through dual GNN
        output = self.gnn(
            x,
            filtered_edge_index_genes,
            filtered_edge_index_patients,
            clinical_features,
            closeness_scores,
            eigenvector_scores,
            betweenness_scores,
        )

        return output
