import torch


def remap_patient_edges(edge_index, batch_patient_ids):
    """
    Remaps patient-patient edges to match batch-local indices.

    - edge_index: Global patient edge index (torch.Tensor [2, num_edges])
    - batch_patient_ids: Global patient IDs present in the batch (torch.Tensor [batch_size])

    Returns:
    - remapped_edge_index: Edge index where all nodes are within the batch range
    """
    # Create mapping from global patient ID -> batch index
    global_to_batch_map = {pid.item(): i for i, pid in enumerate(batch_patient_ids)}

    # Keep only edges where both patients exist in the batch
    mask = [
        (global_to_batch_map[src], global_to_batch_map[dst])
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist())
        if src in global_to_batch_map and dst in global_to_batch_map
    ]

    if not mask:  # No valid edges remaining
        return torch.empty((2, 0), dtype=torch.long)

    # Convert to tensor
    remapped_edge_index = torch.tensor(
        mask, dtype=torch.long
    ).T  # Convert list back to PyTorch tensor

    return remapped_edge_index
