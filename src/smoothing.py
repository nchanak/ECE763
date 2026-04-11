import math
import torch


def dropout_edges(edge_index: torch.Tensor, p_drop: float) -> torch.Tensor:
    """
    Randomly drop edges from edge_index with probability p_drop.
    Assumes edge_index is shape [2, E].
    """
    if p_drop <= 0.0:
        return edge_index

    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > p_drop

    if keep_mask.sum() == 0:
        rand_idx = torch.randint(0, num_edges, (1,), device=edge_index.device)
        keep_mask[rand_idx] = True

    return edge_index[:, keep_mask]


@torch.no_grad()
def smoothed_predict(model, data, num_samples=1000, p_drop=0.1, batch_size=100):
    """
    Monte Carlo smoothed prediction over the whole graph.
    Returns:
      vote_counts: [num_nodes, num_classes]
      smoothed_pred: [num_nodes]
    """
    model.eval()
    device = data.x.device

    clean_out = model(data.x, data.edge_index)
    num_nodes = clean_out.size(0)
    num_classes = clean_out.size(1)

    vote_counts = torch.zeros(num_nodes, num_classes, device=device, dtype=torch.long)

    for start in range(0, num_samples, batch_size):
        cur_batch = min(batch_size, num_samples - start)

        for _ in range(cur_batch):
            noisy_edge_index = dropout_edges(data.edge_index, p_drop)
            out = model(data.x, noisy_edge_index)
            pred = out.argmax(dim=1)
            vote_counts[torch.arange(num_nodes, device=device), pred] += 1

    smoothed_pred = vote_counts.argmax(dim=1)
    return vote_counts, smoothed_pred

import torch


def dropout_edges(edge_index: torch.Tensor, p_drop: float) -> torch.Tensor:
    if p_drop <= 0.0:
        return edge_index

    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > p_drop

    if keep_mask.sum() == 0:
        keep_mask[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True

    return edge_index[:, keep_mask]


@torch.no_grad()
def smoothed_predict_with_edge_index(model, x, edge_index, num_samples=500, p_drop=0.1, batch_size=50):
    model.eval()
    device = x.device

    clean_out = model(x, edge_index)
    num_nodes = clean_out.size(0)
    num_classes = clean_out.size(1)

    vote_counts = torch.zeros(num_nodes, num_classes, dtype=torch.long, device=device)

    for start in range(0, num_samples, batch_size):
        cur_batch = min(batch_size, num_samples - start)

        for _ in range(cur_batch):
            noisy_edge_index = dropout_edges(edge_index, p_drop)
            out = model(x, noisy_edge_index)
            pred = out.argmax(dim=1)
            vote_counts[torch.arange(num_nodes, device=device), pred] += 1

    smoothed_pred = vote_counts.argmax(dim=1)
    return vote_counts, smoothed_pred

@torch.no_grad()
def certify_node_from_votes(vote_counts: torch.Tensor, node_idx: int):
    """
    This is not the full Wang  certificate,
    """
    votes = vote_counts[node_idx]
    sorted_votes, sorted_classes = torch.sort(votes, descending=True)

    top_class = int(sorted_classes[0].item())
    nA = int(sorted_votes[0].item())
    nB = int(sorted_votes[1].item()) if len(sorted_votes) > 1 else 0
    total = int(votes.sum().item())

    pA_hat = nA / total if total > 0 else 0.0
    pB_hat = nB / total if total > 0 else 0.0
    margin = pA_hat - pB_hat

    return {
        "top_class": top_class,
        "nA": nA,
        "nB": nB,
        "total": total,
        "pA_hat": pA_hat,
        "pB_hat": pB_hat,
        "margin": margin,
    }