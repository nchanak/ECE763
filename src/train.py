import torch
import torch.nn.functional as F


def _compute_split_metrics(pred, labels, masks):
    metrics = {}
    for split_name, mask in masks.items():
        correct = (pred[mask] == labels[mask]).sum().item()
        total = int(mask.sum())
        metrics[split_name] = correct / total if total > 0 else 0.0
    return metrics


def train_one_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    metrics = _compute_split_metrics(
        pred=pred,
        labels=data.y,
        masks={
            "train": data.train_mask,
            "val": data.val_mask,
            "test": data.test_mask,
        },
    )
    return metrics, pred


@torch.no_grad()
def evaluate_with_edge_index(model, data, edge_index, edge_weight=None):
    model.eval()
    out = model(data.x, edge_index, edge_weight=edge_weight)
    pred = out.argmax(dim=1)

    metrics = _compute_split_metrics(
        pred=pred,
        labels=data.y,
        masks={
            "train": data.train_mask,
            "val": data.val_mask,
            "test": data.test_mask,
        },
    )
    return metrics, pred


@torch.no_grad()
def evaluate_smoothed(
    model,
    data,
    num_samples=1000,
    batch_size=100,
    mode="edge-drop",
    p_delete=0.1,
    p_add=0.0,
    max_additions=20000,
    certificate_beta=None,
    certificate_alpha=0.001,
    certificate_max_radius=50,
):
    from src.smoothing import smoothed_predict, summarize_certificates

    vote_counts, smoothed_pred = smoothed_predict(
        model=model,
        data=data,
        num_samples=num_samples,
        batch_size=batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )

    metrics = _compute_split_metrics(
        pred=smoothed_pred,
        labels=data.y,
        masks={
            "train": data.train_mask,
            "val": data.val_mask,
            "test": data.test_mask,
        },
    )

    certificate_summary = None
    if certificate_beta is not None:
        certificate_summary, _ = summarize_certificates(
            vote_counts=vote_counts,
            true_labels=data.y,
            mask=data.test_mask,
            alpha=certificate_alpha,
            beta=certificate_beta,
            max_radius=certificate_max_radius,
        )

    return metrics, vote_counts, smoothed_pred, certificate_summary


@torch.no_grad()
def evaluate_smoothed_with_edge_index(
    model,
    data,
    edge_index,
    num_samples=500,
    batch_size=50,
    mode="edge-drop",
    p_delete=0.1,
    p_add=0.0,
    max_additions=20000,
    certificate_beta=None,
    certificate_alpha=0.001,
    certificate_max_radius=50,
):
    from src.smoothing import smoothed_predict_with_edge_index, summarize_certificates

    vote_counts, smoothed_pred = smoothed_predict_with_edge_index(
        model=model,
        x=data.x,
        edge_index=edge_index,
        num_samples=num_samples,
        batch_size=batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )

    metrics = _compute_split_metrics(
        pred=smoothed_pred,
        labels=data.y,
        masks={
            "train": data.train_mask,
            "val": data.val_mask,
            "test": data.test_mask,
        },
    )

    certificate_summary = None
    if certificate_beta is not None:
        certificate_summary, _ = summarize_certificates(
            vote_counts=vote_counts,
            true_labels=data.y,
            mask=data.test_mask,
            alpha=certificate_alpha,
            beta=certificate_beta,
            max_radius=certificate_max_radius,
        )

    return metrics, vote_counts, smoothed_pred, certificate_summary


@torch.no_grad()
def evaluate_smoothed_node_with_edge_index(
    model,
    data,
    edge_index,
    node_idx,
    num_samples=1000,
    mode="symmetric-edge-flip",
    p_delete=0.1,
    p_add=0.0,
    max_additions=20000,
    certificate_beta=None,
    certificate_alpha=0.001,
    certificate_max_radius=50,
):
    from src.smoothing import certify_node_from_votes, smoothed_predict_node

    vote_counts, smoothed_pred = smoothed_predict_node(
        model=model,
        x=data.x,
        edge_index=edge_index,
        node_idx=node_idx,
        num_samples=num_samples,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )

    certificate = certify_node_from_votes(
        vote_counts=vote_counts,
        alpha=certificate_alpha,
        beta=certificate_beta,
        max_radius=certificate_max_radius,
    )
    certificate["predicted_class"] = smoothed_pred
    certificate["true_label"] = int(data.y[node_idx].item())
    certificate["is_correct"] = smoothed_pred == certificate["true_label"]
    certificate["reported_certified_radius"] = (
        int(certificate["certified_radius"])
        if certificate["is_correct"] and certificate["certified_radius"] is not None
        else 0
    )
    certificate["reported_runner_up_certified_radius"] = (
        int(certificate["runner_up_certified_radius"])
        if certificate["is_correct"] and certificate["runner_up_certified_radius"] is not None
        else 0
    )

    return certificate, vote_counts, smoothed_pred
