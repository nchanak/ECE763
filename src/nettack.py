import numpy as np
import scipy.sparse as sp
import torch

from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN as DRGCN

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

    clean_pred, _ = predict_node(target_model, data, data.edge_index, target_node)
    attacked_pred, _ = predict_node(target_model, data, attacked_edge_index, target_node)

    return {
        "target_node": int(target_node),
        "true_label": int(data.y[target_node].item()),
        "clean_pred": int(clean_pred),
        "attacked_pred": int(attacked_pred),
        "success": int(attacked_pred) != int(data.y[target_node].item()),
        "perturbations": attack_model.structure_perturbations,
        "attacked_edge_index": attacked_edge_index,
    }