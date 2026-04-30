import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

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



def _batched_edge_index_from_pairs_batch(
    sampled_pairs_batch: Sequence[torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    if not sampled_pairs_batch:
        raise ValueError("sampled_pairs_batch must contain at least one sample")

    batched_edge_indices = []
    for batch_idx, sampled_pairs in enumerate(sampled_pairs_batch):
        edge_index = _pairs_to_edge_index(sampled_pairs)
        if batch_idx > 0:
            edge_index = edge_index + (batch_idx * num_nodes)
        batched_edge_indices.append(edge_index)

    return torch.cat(batched_edge_indices, dim=1)


def _supports_cuda_batched_forward(model) -> bool:
    model_name = type(model).__name__.lower()
    return "sage" not in model_name

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


def _prepare_target_node_sampler(
    unique_pairs: torch.Tensor,
    num_nodes: int,
    target_node: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = unique_pairs.device
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
    return base_pairs, candidates, current_neighbors


def _sample_prepared_target_node_pairs(
    base_pairs: torch.Tensor,
    candidates: torch.Tensor,
    current_neighbors: torch.Tensor,
    target_node: int,
    p_delete: float,
    p_add: float,
    reference_pairs: torch.Tensor,
) -> torch.Tensor:
    device = base_pairs.device
    random_draws = torch.rand(candidates.size(0), device=device)

    keep_existing = current_neighbors & (random_draws > p_delete)
    add_missing = (~current_neighbors) & (random_draws < p_add)
    selected = candidates[keep_existing | add_missing]

    if selected.numel() == 0:
        sampled_incident = base_pairs.new_empty((2, 0))
    else:
        anchor = torch.full_like(selected, target_node)
        sampled_incident = torch.stack(
            [torch.minimum(anchor, selected), torch.maximum(anchor, selected)],
            dim=0,
        )

    sampled_pairs = torch.cat([base_pairs, sampled_incident], dim=1)
    return _ensure_nonempty_pairs(sampled_pairs, reference_pairs)


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
    total_non_edges = total_possible - unique_pairs.size(1)
    num_additions = _sample_addition_count(total_non_edges, p_add, max_additions)
    addition_pairs = _sample_absent_pairs(
        num_nodes=num_nodes,
        existing_ids=_pair_ids(unique_pairs, num_nodes),
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
    target_sampler: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if unique_pairs.size(1) == 0:
        return unique_pairs

    if target_sampler is None:
        target_sampler = _prepare_target_node_sampler(unique_pairs, num_nodes, target_node)

    return _sample_prepared_target_node_pairs(
        base_pairs=target_sampler[0],
        candidates=target_sampler[1],
        current_neighbors=target_sampler[2],
        target_node=target_node,
        p_delete=p_delete,
        p_add=p_add,
        reference_pairs=unique_pairs,
    )


def _sample_smoothed_pairs(
    unique_pairs: torch.Tensor,
    num_nodes: int,
    mode: str = "edge-drop",
    p_delete: float = 0.1,
    p_add: float = 0.0,
    target_node: Optional[int] = None,
    max_additions: int = 20000,
    target_sampler: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if mode not in {"edge-drop", "sparse-edge-flip", "symmetric-edge-flip"}:
        raise ValueError(f"Unsupported smoothing mode: {mode}")

    if mode == "edge-drop":
        p_add = 0.0
    elif mode == "symmetric-edge-flip":
        p_add = p_delete

    if unique_pairs.size(1) == 0:
        return unique_pairs

    if target_node is not None:
        return _sample_target_node_pairs(
            unique_pairs=unique_pairs,
            num_nodes=num_nodes,
            target_node=target_node,
            p_delete=p_delete,
            p_add=p_add,
            target_sampler=target_sampler,
        )

    return _sample_global_pairs(
        unique_pairs=unique_pairs,
        num_nodes=num_nodes,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )


def sample_smoothed_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    mode: str = "edge-drop",
    p_delete: float = 0.1,
    p_add: float = 0.0,
    target_node: Optional[int] = None,
    max_additions: int = 20000,
) -> torch.Tensor:
    unique_pairs = _unique_undirected_pairs(edge_index)
    if unique_pairs.size(1) == 0:
        return edge_index

    sampled_pairs = _sample_smoothed_pairs(
        unique_pairs=unique_pairs,
        num_nodes=num_nodes,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        target_node=target_node,
        max_additions=max_additions,
    )

    return _pairs_to_edge_index(sampled_pairs)


@torch.no_grad()
def collect_smoothed_vote_counts_with_edge_index(
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
    unique_pairs = _unique_undirected_pairs(edge_index)
    target_sampler = None
    if target_node is not None and unique_pairs.size(1) > 0:
        target_sampler = _prepare_target_node_sampler(unique_pairs, num_nodes, target_node)

    if target_node is None:
        vote_counts = torch.zeros(num_nodes, num_classes, dtype=torch.long, device=device)
    else:
        vote_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    use_batched_forward = device.type == "cuda" and batch_size > 1 and _supports_cuda_batched_forward(model)
    batched_x_cache: Dict[int, torch.Tensor] = {}

    for start in range(0, num_samples, batch_size):
        cur_batch = min(batch_size, num_samples - start)
        if not use_batched_forward or cur_batch == 1:
            for _ in range(cur_batch):
                sampled_pairs = _sample_smoothed_pairs(
                    unique_pairs=unique_pairs,
                    num_nodes=num_nodes,
                    mode=mode,
                    p_delete=p_delete,
                    p_add=p_add,
                    target_node=target_node,
                    max_additions=max_additions,
                    target_sampler=target_sampler,
                )
                noisy_edge_index = _pairs_to_edge_index(sampled_pairs)
                out = model(x, noisy_edge_index)

                if target_node is None:
                    pred = out.argmax(dim=1)
                    vote_counts[torch.arange(num_nodes, device=device), pred] += 1
                else:
                    pred = int(out[target_node].argmax().item())
                    vote_counts[pred] += 1
            continue

        sampled_pairs_batch = []
        for _ in range(cur_batch):
            sampled_pairs_batch.append(
                _sample_smoothed_pairs(
                    unique_pairs=unique_pairs,
                    num_nodes=num_nodes,
                    mode=mode,
                    p_delete=p_delete,
                    p_add=p_add,
                    target_node=target_node,
                    max_additions=max_additions,
                    target_sampler=target_sampler,
                )
            )

        if cur_batch == 1:
            batched_x = x
        else:
            batched_x = batched_x_cache.get(cur_batch)
            if batched_x is None:
                batched_x = x.repeat(cur_batch, 1)
                batched_x_cache[cur_batch] = batched_x

        batched_edge_index = _batched_edge_index_from_pairs_batch(
            sampled_pairs_batch=sampled_pairs_batch,
            num_nodes=num_nodes,
        )
        out = model(batched_x, batched_edge_index).reshape(cur_batch, num_nodes, num_classes)

        if target_node is None:
            pred = out.argmax(dim=2)
            vote_counts += torch.nn.functional.one_hot(pred, num_classes=num_classes).sum(dim=0).to(vote_counts.dtype)
        else:
            pred = out[:, target_node, :].argmax(dim=1)
            vote_counts += torch.bincount(pred, minlength=num_classes)

    return vote_counts


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
    vote_counts = collect_smoothed_vote_counts_with_edge_index(
        model=model,
        x=x,
        edge_index=edge_index,
        num_samples=num_samples,
        batch_size=batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        target_node=target_node,
        max_additions=max_additions,
    )

    if target_node is not None:
        return vote_counts, int(vote_counts.argmax().item())

    smoothed_pred = vote_counts.argmax(dim=1)
    return vote_counts, smoothed_pred


@torch.no_grad()
def smoothed_predict_node(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_idx: int,
    num_samples: int = 1000,
    batch_size: int = 50,
    mode: str = "symmetric-edge-flip",
    p_delete: float = 0.1,
    p_add: float = 0.0,
    max_additions: int = 20000,
):
    vote_counts = collect_smoothed_vote_counts_with_edge_index(
        model=model,
        x=x,
        edge_index=edge_index,
        num_samples=num_samples,
        batch_size=batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        target_node=node_idx,
        max_additions=max_additions,
    )

    smoothed_pred = int(torch.argmax(vote_counts).item())
    return vote_counts, smoothed_pred


def target_node_pair_counts(edge_index: torch.Tensor, num_nodes: int, target_node: int) -> Tuple[int, int]:
    unique_pairs = _unique_undirected_pairs(edge_index)
    if unique_pairs.numel() == 0:
        return 0, max(num_nodes - 1, 0)

    incident_mask = (unique_pairs[0] == target_node) | (unique_pairs[1] == target_node)
    num_present = int(incident_mask.sum().item())
    num_absent = max(num_nodes - 1 - num_present, 0)
    return num_present, num_absent


@lru_cache(maxsize=None)
def _region_masses_for_radius(radius: int, beta: float) -> Tuple[Tuple[float, float, float], ...]:
    q = 1.0 - beta
    masses: List[Tuple[float, float, float]] = []

    for matched_adv_bits in range(radius + 1):
        coefficient = math.comb(radius, matched_adv_bits)
        prob_x = coefficient * (q ** matched_adv_bits) * (beta ** (radius - matched_adv_bits))
        prob_y = coefficient * (beta ** matched_adv_bits) * (q ** (radius - matched_adv_bits))
        ratio = float("inf") if prob_y == 0.0 else prob_x / prob_y
        masses.append((ratio, prob_x, prob_y))

    masses.sort(key=lambda item: item[0], reverse=True)
    return tuple(masses)


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
            if not descending:
                transferred += mass_y
            continue

        take_x = min(remaining, mass_x)
        transferred += (take_x / mass_x) * mass_y
        remaining -= take_x

    return transferred


def _probability_ratio(prob_x: float, prob_y: float) -> float:
    if prob_x <= 0.0 and prob_y <= 0.0:
        return 1.0
    if prob_y <= 0.0:
        return float("inf")
    return prob_x / prob_y


def _binomial_probability(trials: int, successes: int, probability: float) -> float:
    return math.comb(trials, successes) * (probability ** successes) * ((1.0 - probability) ** (trials - successes))


def _asymmetric_region_masses(
    delete_budget: int,
    add_budget: int,
    p_delete: float,
    p_add: float,
) -> List[Tuple[float, float, float]]:
    masses: List[Tuple[float, float, float]] = []

    for preserved_deletes in range(delete_budget + 1):
        prob_x_delete = _binomial_probability(delete_budget, preserved_deletes, 1.0 - p_delete)
        prob_y_delete = _binomial_probability(delete_budget, preserved_deletes, p_add)

        for induced_adds in range(add_budget + 1):
            prob_x_add = _binomial_probability(add_budget, induced_adds, p_add)
            prob_y_add = _binomial_probability(add_budget, induced_adds, 1.0 - p_delete)

            prob_x = prob_x_delete * prob_x_add
            prob_y = prob_y_delete * prob_y_add
            masses.append((_probability_ratio(prob_x, prob_y), prob_x, prob_y))

    masses.sort(key=lambda item: item[0], reverse=True)
    return masses


def certify_radius_from_bounds(
    p_lower: float,
    p_upper: float,
    beta: float,
    max_radius: int = 50,
) -> int:
    """Return the largest symmetric certified radius.

    `beta` is treated as the per-edge flip probability so callers can pass the
    same `p_flip` value used by symmetric smoothing. The current construction is
    invariant under swapping `beta` and `1 - beta`, but keeping the flip-rate
    convention matches the rest of the API and avoids future ambiguity.
    """
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


def certify_asymmetric_budget_from_bounds(
    p_lower: float,
    p_upper: float,
    p_delete: float,
    p_add: float,
    delete_budget: int,
    add_budget: int,
) -> bool:
    masses = _asymmetric_region_masses(
        delete_budget=delete_budget,
        add_budget=add_budget,
        p_delete=p_delete,
        p_add=p_add,
    )
    lower_y = _transfer_probability(masses, p_lower, descending=True)
    upper_y = _transfer_probability(masses, p_upper, descending=False)
    return lower_y > upper_y


def certify_asymmetric_radius_from_bounds(
    p_lower: float,
    p_upper: float,
    p_delete: float,
    p_add: float,
    num_present: Optional[int] = None,
    num_absent: Optional[int] = None,
    max_radius: int = 50,
    max_delete: Optional[int] = None,
    max_add: Optional[int] = None,
) -> Dict[str, object]:
    if num_present is None:
        num_present = max_radius
    if num_absent is None:
        num_absent = max_radius

    delete_limit = min(max_radius, num_present)
    add_limit = min(max_radius, num_absent)
    if max_delete is not None:
        delete_limit = min(delete_limit, max_delete)
    if max_add is not None:
        add_limit = min(add_limit, max_add)

    total_radius = 0
    max_delete_budget = 0
    max_add_budget = 0
    certified_budgets = []

    for delete_budget in range(delete_limit + 1):
        for add_budget in range(add_limit + 1):
            if delete_budget + add_budget > max_radius:
                continue
            if not certify_asymmetric_budget_from_bounds(
                p_lower=p_lower,
                p_upper=p_upper,
                p_delete=p_delete,
                p_add=p_add,
                delete_budget=delete_budget,
                add_budget=add_budget,
            ):
                continue

            certified_budgets.append((delete_budget, add_budget))
            total_radius = max(total_radius, delete_budget + add_budget)
            if add_budget == 0:
                max_delete_budget = max(max_delete_budget, delete_budget)
            if delete_budget == 0:
                max_add_budget = max(max_add_budget, add_budget)

    return {
        "total_radius": int(total_radius),
        "max_delete_budget": int(max_delete_budget),
        "max_add_budget": int(max_add_budget),
        "certified_budgets": certified_budgets,
    }


def certify_node_from_votes(
    vote_counts: torch.Tensor,
    node_idx: Optional[int] = None,
    predicted_class: Optional[int] = None,
    alpha: float = 0.001,
    beta: Optional[float] = None,
    p_delete: Optional[float] = None,
    p_add: Optional[float] = None,
    num_present: Optional[int] = None,
    num_absent: Optional[int] = None,
    max_radius: int = 50,
    max_delete: Optional[int] = None,
    max_add: Optional[int] = None,
) -> Dict[str, float]:
    if vote_counts.dim() == 2:
        if node_idx is None:
            raise ValueError("node_idx is required when vote_counts is 2D")
        votes = vote_counts[node_idx]
    else:
        votes = vote_counts

    sorted_votes, sorted_classes = torch.sort(votes, descending=True)
    certification_top_class = int(sorted_classes[0].item())

    top_class = int(predicted_class) if predicted_class is not None else certification_top_class
    n_a = int(votes[top_class].item())

    other_classes = torch.arange(votes.numel(), device=votes.device)
    other_classes = other_classes[other_classes != top_class]
    if other_classes.numel() > 0:
        other_votes = votes[other_classes]
        best_other_idx = int(torch.argmax(other_votes).item())
        runner_up_class = int(other_classes[best_other_idx].item())
        n_b = int(other_votes[best_other_idx].item())
    else:
        runner_up_class = top_class
        n_b = 0

    total = int(votes.sum().item())

    p_a_hat = n_a / total if total > 0 else 0.0
    p_b_hat = n_b / total if total > 0 else 0.0
    alpha_tail = alpha / 2.0
    p_a_lower = _one_sided_clopper_pearson_lower(n_a, total, alpha_tail)
    p_b_upper = _one_sided_clopper_pearson_upper(n_b, total, alpha_tail)
    p_rest_upper = max(0.0, 1.0 - p_a_lower)
    top2_upper = min(p_b_upper, p_rest_upper)
    abstained = p_a_lower <= p_rest_upper
    top2_abstained = p_a_lower <= top2_upper

    certified_radius = None
    runner_up_radius = None
    certificate_kind = None
    asymmetric_certificate = None
    runner_up_asymmetric_certificate = None
    if p_delete is not None and p_add is not None and num_present is not None and num_absent is not None:
        certificate_kind = "asymmetric"
    elif beta is not None:
        certificate_kind = "symmetric"

    if certificate_kind == "asymmetric":
        if not abstained:
            asymmetric_certificate = certify_asymmetric_radius_from_bounds(
                p_lower=p_a_lower,
                p_upper=p_rest_upper,
                p_delete=p_delete,
                p_add=p_add,
                num_present=num_present,
                num_absent=num_absent,
                max_radius=max_radius,
                max_delete=max_delete,
                max_add=max_add,
            )
            certified_radius = int(asymmetric_certificate["total_radius"])

        if not top2_abstained:
            runner_up_asymmetric_certificate = certify_asymmetric_radius_from_bounds(
                p_lower=p_a_lower,
                p_upper=top2_upper,
                p_delete=p_delete,
                p_add=p_add,
                num_present=num_present,
                num_absent=num_absent,
                max_radius=max_radius,
                max_delete=max_delete,
                max_add=max_add,
            )
            runner_up_radius = int(runner_up_asymmetric_certificate["total_radius"])
    elif certificate_kind == "symmetric":
        if not abstained:
            certified_radius = certify_radius_from_bounds(
                p_lower=p_a_lower,
                p_upper=p_rest_upper,
                beta=beta,
                max_radius=max_radius,
            )
        if not top2_abstained:
            runner_up_radius = certify_radius_from_bounds(
                p_lower=p_a_lower,
                p_upper=top2_upper,
                beta=beta,
                max_radius=max_radius,
            )

    return {
        "top_class": top_class,
        "certification_top_class": certification_top_class,
        "runner_up_class": runner_up_class,
        "nA": n_a,
        "nB": n_b,
        "total": total,
        "pA_hat": p_a_hat,
        "pB_hat": p_b_hat,
        "pA_lower": p_a_lower,
        "pB_upper": p_b_upper,
        "p_rest_upper": p_rest_upper,
        "top2_upper": top2_upper,
        "margin": p_a_hat - p_b_hat,
        "lower_margin": p_a_lower - p_rest_upper,
        "top2_lower_margin": p_a_lower - top2_upper,
        "abstained": abstained,
        "top2_abstained": top2_abstained,
        "certificate_kind": certificate_kind,
        "confidence_level": 1.0 - alpha,
        "certified_radius": certified_radius,
        "runner_up_certified_radius": runner_up_radius,
        "asymmetric_certificate": asymmetric_certificate,
        "runner_up_asymmetric_certificate": runner_up_asymmetric_certificate,
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
            bool(row.get("is_correct", False))
            and not bool(row.get("abstained", False))
            and row.get("certified_radius") is not None
            and int(row["certified_radius"]) >= radius
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
    predicted_labels: Optional[torch.Tensor] = None,
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
    abstained_nodes = 0
    non_abstain_correct_nodes = 0

    for node_idx in node_indices.tolist():
        certificate = certify_node_from_votes(
            vote_counts=vote_counts,
            node_idx=node_idx,
            predicted_class=(int(predicted_labels[node_idx].item()) if predicted_labels is not None else None),
            alpha=alpha,
            beta=beta,
            max_radius=max_radius,
        )
        certificate["node_idx"] = node_idx
        certificate["predicted_class"] = int(certificate["top_class"])
        if certificate["abstained"]:
            abstained_nodes += 1

        if true_labels is not None:
            true_label = int(true_labels[node_idx].item())
            is_correct = certificate["predicted_class"] == true_label
            certificate["true_label"] = true_label
            certificate["is_correct"] = is_correct
            certificate["reported_certified_radius"] = (
                int(certificate["certified_radius"])
                if is_correct and not certificate["abstained"] and certificate["certified_radius"] is not None
                else 0
            )
            certificate["reported_runner_up_certified_radius"] = (
                int(certificate["runner_up_certified_radius"])
                if is_correct and not certificate["abstained"] and certificate["runner_up_certified_radius"] is not None
                else 0
            )
            if is_correct:
                correct_nodes += 1
                if not certificate["abstained"]:
                    non_abstain_correct_nodes += 1
        else:
            is_correct = False
            certificate["reported_certified_radius"] = int(certificate["certified_radius"] or 0)
            certificate["reported_runner_up_certified_radius"] = int(certificate["runner_up_certified_radius"] or 0)

        if not certificate["abstained"] and certificate["certified_radius"] is not None:
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
        "abstained_nodes": abstained_nodes,
        "abstained_fraction": abstained_nodes / total_nodes if total_nodes > 0 else 0.0,
        "non_abstain_accuracy": non_abstain_correct_nodes / max(total_nodes - abstained_nodes, 1),
        "certified_fraction": certified_fraction,
        "positive_certified_accuracy": certified_accuracy_positive,
        "certified_fraction_on_correct": certified_fraction_on_correct,
        "mean_certified_radius": mean_radius,
        "mean_certified_radius_on_correct": mean_radius_on_correct,
        "certified_accuracy_curve": certified_accuracy_curve,
    }
    return summary, per_node
