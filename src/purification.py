import torch


def _unique_undirected_pairs(edge_index):
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0))

    row, col = edge_index
    mask = row < col
    if not torch.any(mask):
        return edge_index.new_empty((2, 0))
    return torch.stack([row[mask], col[mask]], dim=0)


def _jaccard_scores_for_pairs(x, undirected_pairs):
    if undirected_pairs.numel() == 0:
        return torch.empty(0, device=x.device, dtype=torch.float32)

    binary_x = x > 0
    left = binary_x[undirected_pairs[0]]
    right = binary_x[undirected_pairs[1]]

    intersection = (left & right).sum(dim=1)
    union = (left | right).sum(dim=1)
    return torch.where(
        union > 0,
        intersection.float() / union.float(),
        torch.zeros_like(intersection, dtype=torch.float32),
    )


@torch.no_grad()
def purify_edge_index_by_jaccard(x, edge_index, threshold=0.0):
    undirected_pairs = _unique_undirected_pairs(edge_index)
    total_edges = int(undirected_pairs.size(1))
    if total_edges == 0:
        return edge_index.new_empty((2, 0)), {
            "threshold": float(threshold),
            "original_undirected_edges": 0,
            "kept_undirected_edges": 0,
            "edge_retention": 0.0,
            "mean_jaccard": 0.0,
        }

    scores = _jaccard_scores_for_pairs(x, undirected_pairs)

    keep_mask = scores >= threshold
    kept_pairs = undirected_pairs[:, keep_mask]
    if kept_pairs.numel() == 0:
        purified_edge_index = edge_index.new_empty((2, 0))
    else:
        purified_edge_index = torch.cat([kept_pairs, kept_pairs.flip(0)], dim=1)

    kept_edges = int(kept_pairs.size(1))
    return purified_edge_index, {
        "threshold": float(threshold),
        "original_undirected_edges": total_edges,
        "kept_undirected_edges": kept_edges,
        "edge_retention": kept_edges / total_edges if total_edges > 0 else 0.0,
        "mean_jaccard": float(scores.mean().item()) if scores.numel() > 0 else 0.0,
    }


@torch.no_grad()
def purify_target_node_edges_by_jaccard(x, edge_index, target_node, threshold=0.0):
    undirected_pairs = _unique_undirected_pairs(edge_index)
    total_edges = int(undirected_pairs.size(1))
    if total_edges == 0:
        return edge_index.new_empty((2, 0)), {
            "threshold": float(threshold),
            "target_node": int(target_node),
            "original_undirected_edges": 0,
            "kept_undirected_edges": 0,
            "edge_retention": 0.0,
            "target_undirected_edges": 0,
            "kept_target_undirected_edges": 0,
            "target_edge_retention": 0.0,
            "mean_target_jaccard": 0.0,
        }

    scores = _jaccard_scores_for_pairs(x, undirected_pairs)
    target_mask = (undirected_pairs[0] == target_node) | (undirected_pairs[1] == target_node)
    keep_mask = (~target_mask) | (scores >= threshold)
    kept_pairs = undirected_pairs[:, keep_mask]

    if kept_pairs.numel() == 0:
        purified_edge_index = edge_index.new_empty((2, 0))
    else:
        purified_edge_index = torch.cat([kept_pairs, kept_pairs.flip(0)], dim=1)

    target_edges = int(target_mask.sum().item())
    kept_target_edges = int((keep_mask & target_mask).sum().item())
    target_scores = scores[target_mask]
    kept_edges = int(kept_pairs.size(1))

    return purified_edge_index, {
        "threshold": float(threshold),
        "target_node": int(target_node),
        "original_undirected_edges": total_edges,
        "kept_undirected_edges": kept_edges,
        "edge_retention": kept_edges / total_edges if total_edges > 0 else 0.0,
        "target_undirected_edges": target_edges,
        "kept_target_undirected_edges": kept_target_edges,
        "target_edge_retention": kept_target_edges / target_edges if target_edges > 0 else 0.0,
        "mean_target_jaccard": float(target_scores.mean().item()) if target_scores.numel() > 0 else 0.0,
    }