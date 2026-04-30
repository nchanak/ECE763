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


def _cosine_scores_for_pairs(x, undirected_pairs):
    if undirected_pairs.numel() == 0:
        return torch.empty(0, device=x.device, dtype=torch.float32)

    normalized_x = torch.nn.functional.normalize(x.float(), p=2.0, dim=1)
    left = normalized_x[undirected_pairs[0]]
    right = normalized_x[undirected_pairs[1]]
    return (left * right).sum(dim=1)


def _pair_scores_for_operator(x, undirected_pairs, operator="jaccard"):
    operator_name = str(operator or "jaccard").lower()
    if operator_name == "jaccard":
        return _jaccard_scores_for_pairs(x, undirected_pairs)
    if operator_name == "cosine":
        return _cosine_scores_for_pairs(x, undirected_pairs)
    raise ValueError(f"Unsupported purification operator: {operator}")


@torch.no_grad()
def purify_edge_index(x, edge_index, threshold=0.0, operator="jaccard"):
    undirected_pairs = _unique_undirected_pairs(edge_index)
    total_edges = int(undirected_pairs.size(1))
    if total_edges == 0:
        return edge_index.new_empty((2, 0)), {
            "operator": str(operator),
            "threshold": float(threshold),
            "original_undirected_edges": 0,
            "kept_undirected_edges": 0,
            "edge_retention": 0.0,
            "mean_score": 0.0,
        }

    scores = _pair_scores_for_operator(x, undirected_pairs, operator=operator)
    keep_mask = scores >= threshold
    kept_pairs = undirected_pairs[:, keep_mask]
    if kept_pairs.numel() == 0:
        purified_edge_index = edge_index.new_empty((2, 0))
    else:
        purified_edge_index = torch.cat([kept_pairs, kept_pairs.flip(0)], dim=1)

    kept_edges = int(kept_pairs.size(1))
    stats = {
        "operator": str(operator),
        "threshold": float(threshold),
        "original_undirected_edges": total_edges,
        "kept_undirected_edges": kept_edges,
        "edge_retention": kept_edges / total_edges if total_edges > 0 else 0.0,
        "mean_score": float(scores.mean().item()) if scores.numel() > 0 else 0.0,
    }
    if str(operator).lower() == "jaccard":
        stats["mean_jaccard"] = stats["mean_score"]
    return purified_edge_index, stats


@torch.no_grad()
def purify_target_node_edges(x, edge_index, target_node, threshold=0.0, operator="jaccard"):
    undirected_pairs = _unique_undirected_pairs(edge_index)
    total_edges = int(undirected_pairs.size(1))
    if total_edges == 0:
        return edge_index.new_empty((2, 0)), {
            "operator": str(operator),
            "threshold": float(threshold),
            "target_node": int(target_node),
            "original_undirected_edges": 0,
            "kept_undirected_edges": 0,
            "edge_retention": 0.0,
            "target_undirected_edges": 0,
            "kept_target_undirected_edges": 0,
            "target_edge_retention": 0.0,
            "mean_target_score": 0.0,
        }

    target_mask = (undirected_pairs[0] == target_node) | (undirected_pairs[1] == target_node)
    target_pairs = undirected_pairs[:, target_mask]
    target_scores = _pair_scores_for_operator(x, target_pairs, operator=operator)

    keep_mask = torch.ones(total_edges, device=undirected_pairs.device, dtype=torch.bool)
    keep_mask[target_mask] = target_scores >= threshold
    kept_pairs = undirected_pairs[:, keep_mask]

    if kept_pairs.numel() == 0:
        purified_edge_index = edge_index.new_empty((2, 0))
    else:
        purified_edge_index = torch.cat([kept_pairs, kept_pairs.flip(0)], dim=1)

    target_edges = int(target_mask.sum().item())
    kept_target_edges = int((keep_mask & target_mask).sum().item())
    kept_edges = int(kept_pairs.size(1))

    stats = {
        "operator": str(operator),
        "threshold": float(threshold),
        "target_node": int(target_node),
        "original_undirected_edges": total_edges,
        "kept_undirected_edges": kept_edges,
        "edge_retention": kept_edges / total_edges if total_edges > 0 else 0.0,
        "target_undirected_edges": target_edges,
        "kept_target_undirected_edges": kept_target_edges,
        "target_edge_retention": kept_target_edges / target_edges if target_edges > 0 else 0.0,
        "mean_target_score": float(target_scores.mean().item()) if target_scores.numel() > 0 else 0.0,
    }
    if str(operator).lower() == "jaccard":
        stats["mean_target_jaccard"] = stats["mean_target_score"]
    return purified_edge_index, stats


@torch.no_grad()
def purify_edge_index_by_jaccard(x, edge_index, threshold=0.0):
    purified_edge_index, stats = purify_edge_index(x, edge_index, threshold=threshold, operator="jaccard")
    stats.setdefault("mean_jaccard", float(stats.get("mean_score", 0.0)))
    return purified_edge_index, stats


@torch.no_grad()
def purify_target_node_edges_by_jaccard(x, edge_index, target_node, threshold=0.0):
    purified_edge_index, stats = purify_target_node_edges(
        x,
        edge_index,
        target_node=target_node,
        threshold=threshold,
        operator="jaccard",
    )
    stats.setdefault("mean_target_jaccard", float(stats.get("mean_target_score", 0.0)))
    return purified_edge_index, stats