from typing import cast

import torch
import torch.nn.functional as F


def _compute_split_metrics(pred, labels, masks):
    metrics = {}
    for split_name, mask in masks.items():
        correct = (pred[mask] == labels[mask]).sum().item()
        total = int(mask.sum())
        metrics[split_name] = correct / total if total > 0 else 0.0
    return metrics


def _compute_split_metrics_with_abstentions(pred, labels, masks, abstained_mask):
    metrics = _compute_split_metrics(pred=pred, labels=labels, masks=masks)

    for split_name, mask in masks.items():
        total = int(mask.sum())
        split_abstained = int((abstained_mask & mask).sum().item())
        non_abstain = total - split_abstained
        correct_non_abstain = int(((pred == labels) & mask & ~abstained_mask).sum().item())

        metrics[split_name] = correct_non_abstain / total if total > 0 else 0.0
        metrics[f"{split_name}_abstention"] = split_abstained / total if total > 0 else 0.0
        metrics[f"{split_name}_non_abstain_accuracy"] = (
            correct_non_abstain / non_abstain if non_abstain > 0 else 0.0
        )

    return metrics


def train_one_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch_with_noise(
    model,
    data,
    optimizer,
    mode="sparse-edge-flip",
    p_delete=0.02,
    p_add=0.0,
    max_additions=20000,
    purification_operator=None,
    purification_threshold=0.0,
):
    from src.smoothing import sample_smoothed_edge_index
    from src.purification import purify_edge_index

    model.train()
    optimizer.zero_grad()

    noisy_edge_index = sample_smoothed_edge_index(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )
    if purification_operator:
        noisy_edge_index, _ = purify_edge_index(
            x=data.x,
            edge_index=noisy_edge_index,
            threshold=purification_threshold,
            operator=purification_operator,
        )
    out = model(data.x, noisy_edge_index)
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
    selection_num_samples=None,
    certification_num_samples=None,
    selection_batch_size=None,
    certification_batch_size=None,
):
    from src.smoothing import smoothed_predict, summarize_certificates

    selection_num_samples = selection_num_samples or num_samples
    selection_batch_size = selection_batch_size or batch_size

    selection_vote_counts, smoothed_pred = smoothed_predict(
        model=model,
        data=data,
        num_samples=selection_num_samples,
        batch_size=selection_batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )

    vote_counts = selection_vote_counts
    abstained_mask = None
    certificate_summary = None

    if certificate_beta is not None:
        certification_num_samples = certification_num_samples or num_samples
        certification_batch_size = certification_batch_size or batch_size
        vote_counts, _ = smoothed_predict(
            model=model,
            data=data,
            num_samples=certification_num_samples,
            batch_size=certification_batch_size,
            mode=mode,
            p_delete=p_delete,
            p_add=p_add,
            max_additions=max_additions,
        )

        _, all_certificate_rows = summarize_certificates(
            vote_counts=vote_counts,
            predicted_labels=smoothed_pred,
            true_labels=data.y,
            alpha=certificate_alpha,
            beta=certificate_beta,
            max_radius=certificate_max_radius,
        )
        abstained_mask = torch.zeros(data.y.size(0), dtype=torch.bool, device=data.y.device)
        for row in all_certificate_rows:
            abstained_mask[row["node_idx"]] = bool(row["abstained"])

        certificate_summary, _ = summarize_certificates(
            vote_counts=vote_counts,
            predicted_labels=smoothed_pred,
            true_labels=data.y,
            mask=data.test_mask,
            alpha=certificate_alpha,
            beta=certificate_beta,
            max_radius=certificate_max_radius,
        )

    masks = {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }
    if abstained_mask is None:
        metrics = _compute_split_metrics(pred=smoothed_pred, labels=data.y, masks=masks)
    else:
        metrics = _compute_split_metrics_with_abstentions(
            pred=smoothed_pred,
            labels=data.y,
            masks=masks,
            abstained_mask=abstained_mask,
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
    selection_num_samples=None,
    certification_num_samples=None,
    selection_batch_size=None,
    certification_batch_size=None,
):
    from src.smoothing import smoothed_predict_with_edge_index, summarize_certificates

    selection_num_samples = selection_num_samples or num_samples
    selection_batch_size = selection_batch_size or batch_size

    selection_vote_counts, smoothed_pred = smoothed_predict_with_edge_index(
        model=model,
        x=data.x,
        edge_index=edge_index,
        num_samples=selection_num_samples,
        batch_size=selection_batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )

    vote_counts = selection_vote_counts
    abstained_mask = None
    certificate_summary = None

    if certificate_beta is not None:
        certification_num_samples = certification_num_samples or num_samples
        certification_batch_size = certification_batch_size or batch_size
        vote_counts, _ = smoothed_predict_with_edge_index(
            model=model,
            x=data.x,
            edge_index=edge_index,
            num_samples=certification_num_samples,
            batch_size=certification_batch_size,
            mode=mode,
            p_delete=p_delete,
            p_add=p_add,
            max_additions=max_additions,
        )

        _, all_certificate_rows = summarize_certificates(
            vote_counts=vote_counts,
            predicted_labels=smoothed_pred,
            true_labels=data.y,
            alpha=certificate_alpha,
            beta=certificate_beta,
            max_radius=certificate_max_radius,
        )
        abstained_mask = torch.zeros(data.y.size(0), dtype=torch.bool, device=data.y.device)
        for row in all_certificate_rows:
            abstained_mask[row["node_idx"]] = bool(row["abstained"])

        certificate_summary, _ = summarize_certificates(
            vote_counts=vote_counts,
            predicted_labels=smoothed_pred,
            true_labels=data.y,
            mask=data.test_mask,
            alpha=certificate_alpha,
            beta=certificate_beta,
            max_radius=certificate_max_radius,
        )

    masks = {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }
    if abstained_mask is None:
        metrics = _compute_split_metrics(pred=smoothed_pred, labels=data.y, masks=masks)
    else:
        metrics = _compute_split_metrics_with_abstentions(
            pred=smoothed_pred,
            labels=data.y,
            masks=masks,
            abstained_mask=abstained_mask,
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
    certificate_p_delete=None,
    certificate_p_add=None,
    certificate_alpha=0.001,
    certificate_max_radius=50,
    selection_num_samples=None,
    certification_num_samples=None,
    selection_batch_size=None,
    certification_batch_size=None,
    certificate_max_delete=None,
    certificate_max_add=None,
    certificate_report_strategy="rest",
):
    from src.smoothing import certify_node_from_votes, smoothed_predict_node, target_node_pair_counts

    selection_num_samples = selection_num_samples or num_samples
    selection_batch_size = selection_batch_size or min(50, selection_num_samples)
    selection_vote_counts, smoothed_pred = smoothed_predict_node(
        model=model,
        x=data.x,
        edge_index=edge_index,
        node_idx=node_idx,
        num_samples=selection_num_samples,
        batch_size=selection_batch_size,
        mode=mode,
        p_delete=p_delete,
        p_add=p_add,
        max_additions=max_additions,
    )

    vote_counts = selection_vote_counts
    if certificate_beta is not None or (certificate_p_delete is not None and certificate_p_add is not None):
        certification_num_samples = certification_num_samples or num_samples
        certification_batch_size = certification_batch_size or min(50, certification_num_samples)
        vote_counts, _ = smoothed_predict_node(
            model=model,
            x=data.x,
            edge_index=edge_index,
            node_idx=node_idx,
            num_samples=certification_num_samples,
            batch_size=certification_batch_size,
            mode=mode,
            p_delete=p_delete,
            p_add=p_add,
            max_additions=max_additions,
        )

    num_present = None
    num_absent = None
    if certificate_p_delete is not None and certificate_p_add is not None:
        num_present, num_absent = target_node_pair_counts(
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            target_node=node_idx,
        )

    certificate = certify_node_from_votes(
        vote_counts=vote_counts,
        predicted_class=smoothed_pred,
        alpha=certificate_alpha,
        beta=certificate_beta,
        p_delete=certificate_p_delete,
        p_add=certificate_p_add,
        num_present=num_present,
        num_absent=num_absent,
        max_radius=certificate_max_radius,
        max_delete=certificate_max_delete,
        max_add=certificate_max_add,
    )
    certificate["predicted_class"] = smoothed_pred
    certificate["true_label"] = int(data.y[node_idx].item())
    certificate["is_correct"] = smoothed_pred == certificate["true_label"]
    report_strategy = str(certificate_report_strategy or "rest")
    if report_strategy not in {"rest", "top2", "runner-up"}:
        raise ValueError("certificate_report_strategy must be one of: rest, top2, runner-up")

    use_top2_report = report_strategy in {"top2", "runner-up"}
    certificate["reported_rest_certified_radius"] = (
        int(certificate["certified_radius"])
        if certificate["is_correct"] and not certificate["abstained"] and certificate["certified_radius"] is not None
        else 0
    )
    certificate["reported_top2_certified_radius"] = (
        int(certificate["runner_up_certified_radius"])
        if certificate["is_correct"]
        and not certificate["top2_abstained"]
        and certificate["runner_up_certified_radius"] is not None
        else 0
    )
    certificate["reported_runner_up_certified_radius"] = int(certificate["reported_top2_certified_radius"])
    certificate_metadata = cast(dict[str, object], certificate)
    certificate_metadata["certificate_report_strategy"] = "top2" if use_top2_report else "rest"
    certificate["reported_certified_radius"] = int(
        certificate["reported_top2_certified_radius"] if use_top2_report else certificate["reported_rest_certified_radius"]
    )
    asymmetric_certificate = cast(dict[str, object] | None, certificate.get("asymmetric_certificate"))
    runner_up_asymmetric_certificate = cast(
        dict[str, object] | None,
        certificate.get("runner_up_asymmetric_certificate"),
    )
    certificate["reported_rest_asymmetric_total_radius"] = (
        int(cast(int, asymmetric_certificate["total_radius"]))
        if certificate["is_correct"] and not certificate["abstained"] and asymmetric_certificate is not None
        else 0
    )
    certificate["reported_rest_asymmetric_delete_budget"] = (
        int(cast(int, asymmetric_certificate["max_delete_budget"]))
        if certificate["is_correct"] and not certificate["abstained"] and asymmetric_certificate is not None
        else 0
    )
    certificate["reported_rest_asymmetric_add_budget"] = (
        int(cast(int, asymmetric_certificate["max_add_budget"]))
        if certificate["is_correct"] and not certificate["abstained"] and asymmetric_certificate is not None
        else 0
    )
    certificate["reported_runner_up_asymmetric_total_radius"] = (
        int(cast(int, runner_up_asymmetric_certificate["total_radius"]))
        if certificate["is_correct"] and not certificate["top2_abstained"] and runner_up_asymmetric_certificate is not None
        else 0
    )
    certificate["reported_top2_asymmetric_total_radius"] = int(
        certificate["reported_runner_up_asymmetric_total_radius"]
    )
    certificate["reported_top2_asymmetric_delete_budget"] = (
        int(cast(int, runner_up_asymmetric_certificate["max_delete_budget"]))
        if certificate["is_correct"] and not certificate["top2_abstained"] and runner_up_asymmetric_certificate is not None
        else 0
    )
    certificate["reported_top2_asymmetric_add_budget"] = (
        int(cast(int, runner_up_asymmetric_certificate["max_add_budget"]))
        if certificate["is_correct"] and not certificate["top2_abstained"] and runner_up_asymmetric_certificate is not None
        else 0
    )
    certificate["reported_asymmetric_total_radius"] = int(
        certificate["reported_top2_asymmetric_total_radius"]
        if use_top2_report
        else certificate["reported_rest_asymmetric_total_radius"]
    )
    certificate["reported_asymmetric_delete_budget"] = int(
        certificate["reported_top2_asymmetric_delete_budget"]
        if use_top2_report
        else certificate["reported_rest_asymmetric_delete_budget"]
    )
    certificate["reported_asymmetric_add_budget"] = int(
        certificate["reported_top2_asymmetric_add_budget"]
        if use_top2_report
        else certificate["reported_rest_asymmetric_add_budget"]
    )

    raw_certified_radius = int(certificate["certified_radius"] or 0)
    certificate["raw_certified_radius"] = raw_certified_radius
    certificate["raw_runner_up_certified_radius"] = int(certificate["runner_up_certified_radius"] or 0)
    raw_top2_radius = int(certificate["runner_up_certified_radius"] or 0)
    certificate["raw_asymmetric_total_radius"] = (
        int(cast(int, asymmetric_certificate["total_radius"]))
        if asymmetric_certificate is not None
        else 0
    )
    certificate["raw_asymmetric_delete_budget"] = (
        int(cast(int, asymmetric_certificate["max_delete_budget"]))
        if asymmetric_certificate is not None
        else 0
    )
    certificate["raw_asymmetric_add_budget"] = (
        int(cast(int, asymmetric_certificate["max_add_budget"]))
        if asymmetric_certificate is not None
        else 0
    )
    certificate["raw_asymmetric_budget_count"] = (
        len(cast(list[object], asymmetric_certificate.get("certified_budgets", [])))
        if asymmetric_certificate is not None
        else 0
    )
    certificate["certificate_attempted"] = bool(certificate.get("certificate_kind"))
    active_abstained = bool(certificate["top2_abstained"] if use_top2_report else certificate["abstained"])
    active_raw_radius = raw_top2_radius if use_top2_report else raw_certified_radius
    certificate["certificate_blocked_by_abstention"] = bool(
        certificate["is_correct"] and active_abstained
    )
    certificate["certificate_zero_radius_without_abstention"] = bool(
        certificate["is_correct"] and not active_abstained and active_raw_radius <= 0
    )
    if not certificate["is_correct"]:
        certificate_metadata["certificate_failure_reason"] = "prediction-incorrect"
    elif active_abstained:
        certificate_metadata["certificate_failure_reason"] = "abstained"
    elif active_raw_radius <= 0:
        certificate_metadata["certificate_failure_reason"] = "zero-radius"
    else:
        certificate_metadata["certificate_failure_reason"] = "positive-radius"

    return certificate, vote_counts, smoothed_pred
