import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from scipy.stats import beta as beta_dist


def _one_sided_clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0 or successes <= 0:
        return 0.0
    return float(beta_dist.ppf(alpha, successes, trials - successes + 1))


def _one_sided_clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    return float(beta_dist.ppf(1.0 - alpha, successes + 1, trials - successes))


def _unique_undirected_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0))

    row, col = edge_index
    mask = row != col
    row = row[mask]
    col = col[mask]

    if row.numel() == 0:
        return edge_index.new_empty((2, 0))

    pairs = torch.stack([torch.minimum(row, col), torch.maximum(row, col)], dim=0)
    return torch.unique(pairs, dim=1)


def _pair_ids(pairs: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return pairs[0].long() * num_nodes + pairs[1].long()


def _pairs_from_ids(pair_ids: Sequence[int], num_nodes: int, device: torch.device) -> torch.Tensor:
    if len(pair_ids) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    ids = torch.tensor(list(pair_ids), dtype=torch.long, device=device)
    row = torch.div(ids, num_nodes, rounding_mode="floor")
    col = ids % num_nodes
    return torch.stack([row, col], dim=0)


def _pairs_to_edge_index(pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return pairs.new_empty((2, 0))

    reverse_pairs = pairs.flip(0)
    return torch.cat([pairs, reverse_pairs], dim=1)


def _ensure_nonempty_pairs(sampled_pairs: torch.Tensor, reference_pairs: torch.Tensor) -> torch.Tensor:
    if sampled_pairs.size(1) > 0 or reference_pairs.size(1) == 0:
        return sampled_pairs

    keep_idx = torch.randint(0, reference_pairs.size(1), (1,), device=reference_pairs.device)
    return reference_pairs[:, keep_idx]


def _sample_addition_count(total_non_edges: int, p_add: float, max_additions: int) -> int:
    if total_non_edges <= 0 or p_add <= 0.0:
        return 0

    sampled = torch.distributions.binomial.Binomial(
        total_count=float(total_non_edges),
        probs=float(p_add),
    ).sample()
    return min(int(sampled.item()), total_non_edges, max_additions)


def _sample_absent_pairs(
    num_nodes: int,
    existing_ids: torch.Tensor,
    num_pairs: int,
    device: torch.device,
) -> torch.Tensor:
    if num_pairs <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    existing = set(int(pair_id) for pair_id in existing_ids.detach().cpu().tolist())
    sampled: set[int] = set()
    total_possible = num_nodes * (num_nodes - 1) // 2
    target = min(num_pairs, total_possible - len(existing))

    attempts = 0
    while len(sampled) < target and attempts < 256:
        batch_size = max((target - len(sampled)) * 4, 64)
        left = torch.randint(0, num_nodes, (batch_size,), device=device)
        right = torch.randint(0, num_nodes - 1, (batch_size,), device=device)
        right = right + (right >= left).long()

        low = torch.minimum(left, right)
        high = torch.maximum(left, right)
        candidate_ids = (low * num_nodes + high).detach().cpu().tolist()

        for pair_id in candidate_ids:
            if pair_id in existing or pair_id in sampled:
                continue
            sampled.add(int(pair_id))
            if len(sampled) == target:
                break

        attempts += 1

    return _pairs_from_ids(sorted(sampled), num_nodes, device)


def _sample_global_pairs(
    unique_pairs: torch.Tensor,
    num_nodes: int,
    p_delete: float,
    p_add: float,
    max_additions: int,
) -> torch.Tensor:
    device = unique_pairs.device
    if unique_pairs.size(1) == 0:
        return unique_pairs

    keep_mask = torch.rand(unique_pairs.size(1), device=device) > p_delete
    kept_pairs = unique_pairs[:, keep_mask]
    kept_pairs = _ensure_nonempty_pairs(kept_pairs, unique_pairs)

    if p_add <= 0.0:
        return kept_pairs

    total_possible = num_nodes * (num_nodes - 1) // 2
    total_non_edges = total_possible - kept_pairs.size(1)
    num_additions = _sample_addition_count(total_non_edges, p_add, max_additions)
    addition_pairs = _sample_absent_pairs(
        num_nodes=num_nodes,
        existing_ids=_pair_ids(kept_pairs, num_nodes),
        num_pairs=num_additions,
        device=device,
    )

    if addition_pairs.numel() == 0:
        return kept_pairs

    combined_pairs = torch.cat([kept_pairs, addition_pairs], dim=1)
    return torch.unique(combined_pairs, dim=1)


def _sample_target_node_pairs(
    unique_pairs: torch.Tensor,
    num_nodes: int,
    target_node: int,
    p_delete: float,
    p_add: float,
) -> torch.Tensor:
    device = unique_pairs.device
    if unique_pairs.size(1) == 0:
        return unique_pairs

    incident_mask = (unique_pairs[0] == target_node) | (unique_pairs[1] == target_node)
    base_pairs = unique_pairs[:, ~incident_mask]
    incident_pairs = unique_pairs[:, incident_mask]

    neighbor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if incident_pairs.size(1) > 0:
        neighbors = torch.where(
            incident_pairs[0] == target_node,
            incident_pairs[1],
            incident_pairs[0],
        )
        neighbor_mask[neighbors] = True

    candidates = torch.arange(num_nodes, device=device)
    candidates = candidates[candidates != target_node]
    current_neighbors = neighbor_mask[candidates]
    random_draws = torch.rand(candidates.size(0), device=device)

    keep_existing = current_neighbors & (random_draws > p_delete)
    add_missing = (~current_neighbors) & (random_draws < p_add)
    selected = candidates[keep_existing | add_missing]

    if selected.numel() == 0:
        sampled_incident = unique_pairs.new_empty((2, 0))
    else:
        anchor = torch.full_like(selected, target_node)
        sampled_incident = torch.stack(
            [torch.minimum(anchor, selected), torch.maximum(anchor, selected)],
            dim=0,
        )

    sampled_pairs = torch.cat([base_pairs, sampled_incident], dim=1)
    return _ensure_nonempty_pairs(sampled_pairs, unique_pairs)


def sample_smoothed_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    mode: str = "edge-drop",
    p_delete: float = 0.1,
    p_add: float = 0.0,
    target_node: Optional[int] = None,
    max_additions: int = 20000,
) -> torch.Tensor:
    if mode not in {"edge-drop", "sparse-edge-flip", "symmetric-edge-flip"}:
        raise ValueError(f"Unsupported smoothing mode: {mode}")

    if mode == "edge-drop":
        p_add = 0.0
    elif mode == "symmetric-edge-flip":
        p_add = p_delete

    unique_pairs = _unique_undirected_pairs(edge_index)
    if unique_pairs.size(1) == 0:
        return edge_index

    if target_node is not None:
        sampled_pairs = _sample_target_node_pairs(
            unique_pairs=unique_pairs,
            num_nodes=num_nodes,
            target_node=target_node,
            p_delete=p_delete,
            p_add=p_add,
        )
    else:
        sampled_pairs = _sample_global_pairs(
            unique_pairs=unique_pairs,
            num_nodes=num_nodes,
            p_delete=p_delete,
            p_add=p_add,
            max_additions=max_additions,
        )

    return _pairs_to_edge_index(sampled_pairs)


@torch.no_grad()
def smoothed_predict(
    model,
    data,
    num_samples: int = 1000,
    batch_size: int = 100,
    mode: str = "edge-drop",
    p_delete: float = 0.1,
    p_add: float = 0.0,
    target_node: Optional[int] = None,
    max_additions: int = 20000,
):
    return smoothed_predict_with_edge_index(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        num_samples=num_samples,
        batch_size=batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        target_node=target_node,
        max_additions=max_additions,
    )


@torch.no_grad()
def smoothed_predict_with_edge_index(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_samples: int = 500,
    batch_size: int = 50,
    mode: str = "edge-drop",
    p_delete: float = 0.1,
    p_add: float = 0.0,
    target_node: Optional[int] = None,
    max_additions: int = 20000,
):
    model.eval()
    device = x.device

    clean_out = model(x, edge_index)
    num_nodes = clean_out.size(0)
    num_classes = clean_out.size(1)

    vote_counts = torch.zeros(num_nodes, num_classes, dtype=torch.long, device=device)

    for start in range(0, num_samples, batch_size):
        cur_batch = min(batch_size, num_samples - start)
        for _ in range(cur_batch):
            noisy_edge_index = sample_smoothed_edge_index(
                edge_index=edge_index,
                num_nodes=num_nodes,
                mode=mode,
                p_delete=p_delete,
                p_add=p_add,
                target_node=target_node,
                max_additions=max_additions,
            )
            out = model(x, noisy_edge_index)
            pred = out.argmax(dim=1)
            vote_counts[torch.arange(num_nodes, device=device), pred] += 1

    smoothed_pred = vote_counts.argmax(dim=1)
    return vote_counts, smoothed_pred


@torch.no_grad()
def smoothed_predict_node(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_idx: int,
    num_samples: int = 1000,
    mode: str = "symmetric-edge-flip",
    p_delete: float = 0.1,
    p_add: float = 0.0,
    max_additions: int = 20000,
):
    model.eval()
    device = x.device
    num_classes = model(x, edge_index).size(1)
    vote_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    for _ in range(num_samples):
        noisy_edge_index = sample_smoothed_edge_index(
            edge_index=edge_index,
            num_nodes=x.size(0),
            mode=mode,
            p_delete=p_delete,
            p_add=p_add,
            target_node=node_idx,
            max_additions=max_additions,
        )
        pred = model(x, noisy_edge_index)[node_idx].argmax().item()
        vote_counts[pred] += 1

    smoothed_pred = int(torch.argmax(vote_counts).item())
    return vote_counts, smoothed_pred


def _region_masses_for_radius(radius: int, beta: float) -> List[Tuple[float, float, float]]:
    q = 1.0 - beta
    masses: List[Tuple[float, float, float]] = []

    for matched_adv_bits in range(radius + 1):
        coefficient = math.comb(radius, matched_adv_bits)
        prob_x = coefficient * (q ** matched_adv_bits) * (beta ** (radius - matched_adv_bits))
        prob_y = coefficient * (beta ** matched_adv_bits) * (q ** (radius - matched_adv_bits))
        ratio = float("inf") if prob_y == 0.0 else prob_x / prob_y
        masses.append((ratio, prob_x, prob_y))

    masses.sort(key=lambda item: item[0], reverse=True)
    return masses


def _transfer_probability(masses: Sequence[Tuple[float, float, float]], x_probability: float, descending: bool) -> float:
    if x_probability <= 0.0:
        return 0.0

    remaining = min(max(x_probability, 0.0), 1.0)
    transferred = 0.0
    ordered_masses = masses if descending else list(reversed(masses))

    for _, mass_x, mass_y in ordered_masses:
        if remaining <= 0.0:
            break
        if mass_x <= 0.0:
            continue

        take_x = min(remaining, mass_x)
        transferred += (take_x / mass_x) * mass_y
        remaining -= take_x

    return transferred


def certify_radius_from_bounds(
    p_lower: float,
    p_upper: float,
    beta: float,
    max_radius: int = 50,
) -> int:
    certified_radius = 0

    for radius in range(max_radius + 1):
        masses = _region_masses_for_radius(radius, beta)
        lower_y = _transfer_probability(masses, p_lower, descending=True)
        upper_y = _transfer_probability(masses, p_upper, descending=False)

        if lower_y > upper_y:
            certified_radius = radius
        else:
            break

    return certified_radius


def certify_node_from_votes(
    vote_counts: torch.Tensor,
    node_idx: Optional[int] = None,
    alpha: float = 0.001,
    beta: Optional[float] = None,
    max_radius: int = 50,
) -> Dict[str, float]:
    if vote_counts.dim() == 2:
        if node_idx is None:
            raise ValueError("node_idx is required when vote_counts is 2D")
        votes = vote_counts[node_idx]
    else:
        votes = vote_counts

    sorted_votes, sorted_classes = torch.sort(votes, descending=True)

    top_class = int(sorted_classes[0].item())
    runner_up_class = int(sorted_classes[1].item()) if len(sorted_classes) > 1 else top_class
    n_a = int(sorted_votes[0].item())
    n_b = int(sorted_votes[1].item()) if len(sorted_votes) > 1 else 0
    total = int(votes.sum().item())

    p_a_hat = n_a / total if total > 0 else 0.0
    p_b_hat = n_b / total if total > 0 else 0.0
    alpha_tail = alpha / 2.0
    p_a_lower = _one_sided_clopper_pearson_lower(n_a, total, alpha_tail)
    p_b_upper = _one_sided_clopper_pearson_upper(n_b, total, alpha_tail)
    p_rest_upper = max(0.0, 1.0 - p_a_lower)

    certified_radius = None
    runner_up_radius = None
    if beta is not None:
        certified_radius = certify_radius_from_bounds(
            p_lower=p_a_lower,
            p_upper=p_rest_upper,
            beta=beta,
            max_radius=max_radius,
        )
        runner_up_radius = certify_radius_from_bounds(
            p_lower=p_a_lower,
            p_upper=min(p_b_upper, p_rest_upper),
            beta=beta,
            max_radius=max_radius,
        )

    return {
        "top_class": top_class,
        "runner_up_class": runner_up_class,
        "nA": n_a,
        "nB": n_b,
        "total": total,
        "pA_hat": p_a_hat,
        "pB_hat": p_b_hat,
        "pA_lower": p_a_lower,
        "pB_upper": p_b_upper,
        "p_rest_upper": p_rest_upper,
        "margin": p_a_hat - p_b_hat,
        "lower_margin": p_a_lower - p_rest_upper,
        "confidence_level": 1.0 - alpha,
        "certified_radius": certified_radius,
        "runner_up_certified_radius": runner_up_radius,
    }


def build_certified_accuracy_curve(
    certificate_rows: Sequence[Dict[str, object]],
    max_radius: Optional[int] = None,
) -> List[Dict[str, float]]:
    if not certificate_rows:
        return []

    if max_radius is None:
        max_radius = 0
        for row in certificate_rows:
            radius = row.get("certified_radius")
            if radius is not None:
                max_radius = max(max_radius, int(radius))

    total_nodes = len(certificate_rows)
    curve = []

    for radius in range(max_radius + 1):
        certified_correct = sum(
            bool(row.get("is_correct", False)) and int(row.get("certified_radius") or 0) >= radius
            for row in certificate_rows
        )
        curve.append(
            {
                "radius": radius,
                "certified_accuracy": certified_correct / total_nodes if total_nodes > 0 else 0.0,
            }
        )

    return curve


def summarize_certificates(
    vote_counts: torch.Tensor,
    true_labels: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.001,
    beta: Optional[float] = None,
    max_radius: int = 50,
):
    if vote_counts.dim() != 2:
        raise ValueError("summarize_certificates expects vote_counts shaped [num_nodes, num_classes]")

    if mask is None:
        node_indices = torch.arange(vote_counts.size(0), device=vote_counts.device)
    else:
        node_indices = mask.nonzero(as_tuple=False).view(-1)

    per_node = []
    certified_radii = []
    correct_radii = []
    correct_nodes = 0

    for node_idx in node_indices.tolist():
        certificate = certify_node_from_votes(
            vote_counts=vote_counts,
            node_idx=node_idx,
            alpha=alpha,
            beta=beta,
            max_radius=max_radius,
        )
        certificate["node_idx"] = node_idx

        if true_labels is not None:
            true_label = int(true_labels[node_idx].item())
            is_correct = certificate["top_class"] == true_label
            certificate["true_label"] = true_label
            certificate["is_correct"] = is_correct
            certificate["reported_certified_radius"] = (
                int(certificate["certified_radius"])
                if is_correct and certificate["certified_radius"] is not None
                else 0
            )
            certificate["reported_runner_up_certified_radius"] = (
                int(certificate["runner_up_certified_radius"])
                if is_correct and certificate["runner_up_certified_radius"] is not None
                else 0
            )
            if is_correct:
                correct_nodes += 1
        else:
            is_correct = False
            certificate["reported_certified_radius"] = int(certificate["certified_radius"] or 0)
            certificate["reported_runner_up_certified_radius"] = int(certificate["runner_up_certified_radius"] or 0)

        if certificate["certified_radius"] is not None:
            certified_radii.append(certificate["certified_radius"])
            if is_correct:
                correct_radii.append(certificate["certified_radius"])

        per_node.append(certificate)

    total_nodes = len(per_node)
    certified_fraction = 0.0
    mean_radius = 0.0
    mean_radius_on_correct = 0.0
    certified_accuracy_positive = 0.0
    certified_fraction_on_correct = 0.0

    if certified_radii:
        certified_fraction = sum(radius > 0 for radius in certified_radii) / len(certified_radii)
        mean_radius = sum(certified_radii) / len(certified_radii)

    if correct_radii:
        mean_radius_on_correct = sum(correct_radii) / len(correct_radii)
        certified_accuracy_positive = sum(radius > 0 for radius in correct_radii) / total_nodes if total_nodes > 0 else 0.0
        certified_fraction_on_correct = sum(radius > 0 for radius in correct_radii) / correct_nodes if correct_nodes > 0 else 0.0

    certified_accuracy_curve = build_certified_accuracy_curve(per_node, max_radius=max_radius)

    summary = {
        "evaluated_nodes": total_nodes,
        "correct_nodes": correct_nodes,
        "correct_fraction": correct_nodes / total_nodes if total_nodes > 0 else 0.0,
        "certified_fraction": certified_fraction,
        "positive_certified_accuracy": certified_accuracy_positive,
        "certified_fraction_on_correct": certified_fraction_on_correct,
        "mean_certified_radius": mean_radius,
        "mean_certified_radius_on_correct": mean_radius_on_correct,
        "certified_accuracy_curve": certified_accuracy_curve,
    }
    return summary, per_node
