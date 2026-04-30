import numpy as np
import scipy.sparse as sp
import torch

from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN as DRGCN

from src.purification import purify_target_node_edges

# Targeted attack, flips prediction with single edge flip
def pyg_to_scipy_adj(data) -> sp.csr_matrix:
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    vals = np.ones(len(row), dtype=np.float32)
    n = data.num_nodes

    adj = sp.csr_matrix((vals, (row, col)), shape=(n, n))
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def pyg_features_to_scipy(data) -> sp.csr_matrix:
    return sp.csr_matrix(data.x.cpu().numpy())


def scipy_adj_to_edge_index(adj: sp.csr_matrix, device):
    adj = adj.tocoo()
    row = torch.from_numpy(adj.row).long()
    col = torch.from_numpy(adj.col).long()
    return torch.stack([row, col], dim=0).to(device)


@torch.no_grad()
def predict_node(model, data, edge_index, node_idx):
    model.eval()
    out = model(data.x, edge_index)
    pred = out.argmax(dim=1)
    return pred[node_idx].item(), out[node_idx].cpu()


@torch.no_grad()
def get_correct_test_node_metadata(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)
    pred = out.argmax(dim=1)
    top_probs, _ = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
    degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes)

    correct = ((pred == data.y) & data.test_mask).nonzero(as_tuple=False).view(-1)
    metadata = []

    for node_idx in correct.cpu().tolist():
        clean_confidence = float(top_probs[node_idx, 0].item())
        runner_up = float(top_probs[node_idx, 1].item()) if top_probs.size(1) > 1 else 0.0
        metadata.append(
            {
                "node_idx": int(node_idx),
                "true_label": int(data.y[node_idx].item()),
                "degree": int(degree[node_idx].item()),
                "clean_confidence": clean_confidence,
                "clean_margin": clean_confidence - runner_up,
            }
        )

    return metadata


def _annotate_bucket_ranks(rows, key, bucket_key, num_buckets=3):
    if not rows:
        return rows

    ordered = sorted(rows, key=lambda row: (row[key], row["node_idx"]))
    total = len(ordered)
    for rank, row in enumerate(ordered):
        row[bucket_key] = min(num_buckets - 1, rank * num_buckets // total)
    return rows


def _choose_nodes_from_metadata(metadata, k=10, strategy="stratified", return_metadata=False):
    if not metadata:
        return ([], {}) if return_metadata else []

    if strategy == "first":
        selected_rows = sorted(metadata, key=lambda row: row["node_idx"])[:k]
    else:
        _annotate_bucket_ranks(metadata, key="degree", bucket_key="degree_bucket")
        _annotate_bucket_ranks(metadata, key="clean_margin", bucket_key="margin_bucket")

        buckets = {}
        for row in metadata:
            bucket_key = (row["degree_bucket"], row["margin_bucket"])
            buckets.setdefault(bucket_key, []).append(row)

        for rows in buckets.values():
            rows.sort(key=lambda row: (row["clean_confidence"], row["degree"], row["node_idx"]))

        selected_rows = []
        bucket_keys = sorted(buckets)
        while len(selected_rows) < min(k, len(metadata)):
            progress = False
            for bucket_key in bucket_keys:
                rows = buckets[bucket_key]
                if not rows:
                    continue
                selected_rows.append(rows.pop(0))
                progress = True
                if len(selected_rows) >= k:
                    break
            if not progress:
                break

        selected_rows.sort(key=lambda row: (row["degree_bucket"], row["margin_bucket"], row["node_idx"]))

    selected_nodes = [row["node_idx"] for row in selected_rows]
    if return_metadata:
        return selected_nodes, {row["node_idx"]: row for row in selected_rows}
    return selected_nodes


def choose_correct_test_nodes(model, data, k=10, strategy="stratified", return_metadata=False):
    metadata = get_correct_test_node_metadata(model, data)
    return _choose_nodes_from_metadata(
        metadata=metadata,
        k=k,
        strategy=strategy,
        return_metadata=return_metadata,
    )


def choose_jointly_correct_test_nodes(
    reference_model,
    comparison_model,
    data,
    k=10,
    strategy="stratified",
    return_metadata=False,
):
    reference_metadata = get_correct_test_node_metadata(reference_model, data)
    comparison_nodes = {
        row["node_idx"]
        for row in get_correct_test_node_metadata(comparison_model, data)
    }
    joint_metadata = [row for row in reference_metadata if row["node_idx"] in comparison_nodes]
    return _choose_nodes_from_metadata(
        metadata=joint_metadata,
        k=k,
        strategy=strategy,
        return_metadata=return_metadata,
    )


def train_deeprobust_surrogate(data, device="cpu"):
    adj = pyg_to_scipy_adj(data)
    features = pyg_features_to_scipy(data)
    labels = data.y.cpu().numpy()

    idx_train = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    idx_val = data.val_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

    surrogate = DRGCN(
        nfeat=features.shape[1],
        nclass=int(labels.max()) + 1,
        nhid=16,
        dropout=0.0,
        with_relu=False,
        with_bias=False,
        device=device,
    ).to(device)

    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate, adj, features, labels


def _run_single_nettack_attack(
    target_model,
    data,
    surrogate,
    adj,
    features,
    labels,
    target_node,
    n_perturbations,
    device="cpu",
):
    attack_model = Nettack(
        surrogate,
        nnodes=adj.shape[0],
        attack_structure=True,
        attack_features=False,
        device=device,
    ).to(device)

    attack_model.attack(
        features,
        adj,
        labels,
        target_node,
        n_perturbations,
        direct=True,
        verbose=False,
    )

    modified_adj = attack_model.modified_adj
    attacked_edge_index = scipy_adj_to_edge_index(modified_adj, device=device)
    attacked_pred, _ = predict_node(target_model, data, attacked_edge_index, target_node)
    return attacked_edge_index, int(attacked_pred), attack_model.structure_perturbations


def run_nettack_on_node(
    target_model,
    data,
    surrogate,
    adj,
    features,
    labels,
    target_node,
    n_perturbations=3,
    device="cpu",
):
    clean_pred, _ = predict_node(target_model, data, data.edge_index, target_node)
    attacked_edge_index, attacked_pred, perturbations = _run_single_nettack_attack(
        target_model=target_model,
        data=data,
        surrogate=surrogate,
        adj=adj,
        features=features,
        labels=labels,
        target_node=target_node,
        n_perturbations=n_perturbations,
        device=device,
    )

    return {
        "target_node": int(target_node),
        "true_label": int(data.y[target_node].item()),
        "clean_pred": int(clean_pred),
        "attacked_pred": int(attacked_pred),
        "success": int(attacked_pred) != int(data.y[target_node].item()),
        "perturbations": perturbations,
        "attacked_edge_index": attacked_edge_index,
    }


def run_purification_aware_nettack_on_node(
    target_model,
    data,
    surrogate,
    adj,
    features,
    labels,
    target_node,
    purification_thresholds,
    purification_operator="jaccard",
    max_perturbations=5,
    require_all_thresholds=True,
    device="cpu",
):
    thresholds = tuple(sorted(float(threshold) for threshold in purification_thresholds))
    if not thresholds:
        raise ValueError("purification_thresholds must be non-empty")

    clean_pred, _ = predict_node(target_model, data, data.edge_index, target_node)
    true_label = int(data.y[target_node].item())
    best_result = None
    best_score = None

    for n_perturbations in range(1, int(max_perturbations) + 1):
        attacked_edge_index, attacked_pred, perturbations = _run_single_nettack_attack(
            target_model=target_model,
            data=data,
            surrogate=surrogate,
            adj=adj,
            features=features,
            labels=labels,
            target_node=target_node,
            n_perturbations=n_perturbations,
            device=device,
        )

        purified_predictions = []
        target_retentions = []
        successful_thresholds = []
        for threshold in thresholds:
            purified_edge_index, purification_stats = purify_target_node_edges(
                x=data.x,
                edge_index=attacked_edge_index,
                target_node=target_node,
                threshold=threshold,
                operator=purification_operator,
            )
            purified_pred, _ = predict_node(
                target_model,
                data,
                purified_edge_index,
                target_node,
            )
            purified_predictions.append(int(purified_pred))
            target_retentions.append(float(purification_stats["target_edge_retention"]))
            successful_thresholds.append(int(purified_pred) != true_label)

        success_count = int(sum(successful_thresholds))
        success = success_count == len(thresholds) if require_all_thresholds else success_count > 0
        score = (int(success), success_count, int(attacked_pred != true_label), -n_perturbations)
        if best_score is None or score > best_score:
            best_score = score
            best_result = {
                "target_node": int(target_node),
                "true_label": int(true_label),
                "clean_pred": int(clean_pred),
                "attacked_pred": int(attacked_pred),
                "success": int(success),
                "baseline_success": int(attacked_pred != true_label),
                "adaptive_success_count": int(success_count),
                "adaptive_success_rate": (success_count / len(thresholds)) if thresholds else 0.0,
                "adaptive_require_all_thresholds": int(require_all_thresholds),
                "adaptive_thresholds": ";".join(f"{threshold:.3f}" for threshold in thresholds),
                "adaptive_purified_predictions": ";".join(str(prediction) for prediction in purified_predictions),
                "adaptive_mean_target_edge_retention": (
                    sum(target_retentions) / len(target_retentions) if target_retentions else 0.0
                ),
                "perturbation_budget": int(n_perturbations),
                "perturbations": perturbations,
                "attacked_edge_index": attacked_edge_index,
            }

    if best_result is None:
        raise RuntimeError("purification-aware Nettack did not evaluate any perturbation budgets")
    return best_result