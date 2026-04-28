import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.attack import run_prbcd_attack
from src.models import GCN
from src.nettack import choose_correct_test_nodes, run_nettack_on_node, train_deeprobust_surrogate
from src.reporting import (
    plot_attack_budget_accuracy,
    plot_certificate_radius,
    plot_certified_accuracy_curves,
    plot_dual_accuracy_curve,
    save_csv_rows,
    save_json_report,
)
from src.smoothing import build_certified_accuracy_curve, certify_node_from_votes
from src.train import (
    evaluate,
    evaluate_smoothed,
    evaluate_smoothed_node_with_edge_index,
    evaluate_smoothed_with_edge_index,
    evaluate_with_edge_index,
    train_one_epoch,
)

GLOBAL_ATTACK_BUDGETS = [5, 10, 20, 50]
EDGE_DROP_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4]
SPARSE_FLIP_SWEEP = [
    {"p_delete": 0.01, "p_add": 0.000005, "max_additions": 64},
    {"p_delete": 0.02, "p_add": 0.000010, "max_additions": 96},
    {"p_delete": 0.05, "p_add": 0.000020, "max_additions": 160},
]
CERTIFIED_ACCURACY_SWEEP = [0.01, 0.02, 0.05, 0.10]
LOCAL_CERTIFICATE_SWEEP = [0.01, 0.02, 0.05, 0.10]
CERTIFIED_ACCURACY_SAMPLES = 100
CERTIFIED_ACCURACY_NODE_COUNT = 10
CONFIDENCE_Z = 1.96
LOCAL_CERTIFICATE_SAMPLES = 400
CERTIFICATE_MAX_RADIUS = 5
ASYMMETRIC_CERTIFICATE_MAX_DELETE = 5
ASYMMETRIC_CERTIFICATE_MAX_ADD = 5
NETTACK_TARGET_COUNT = 5
FOCUS_SELECTION_P_FLIP = 0.02
FOCUS_SELECTION_SAMPLES = 100
DEFAULT_SEEDS = [42]
DEFAULT_SMOOTHING_MODES = ["edge-drop", "sparse-edge-flip", "symmetric-cert", "sparse-asymmetric-cert"]


def _parse_int_list(value: str):
    if not value or value.lower() in {"none", "skip"}:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str):
    if not value or value.lower() in {"none", "skip"}:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_seed_list(value: str):
    return _parse_int_list(value)


def _parse_sparse_flip_sweep(value: str):
    configs = []
    if not value or value.lower() in {"none", "skip"}:
        return configs

    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Sparse flip configs must look like p_delete:p_add:max_additions;..."
            )
        configs.append(
            {
                "p_delete": float(parts[0]),
                "p_add": float(parts[1]),
                "max_additions": int(parts[2]),
            }
        )
    return configs


def _parse_optional_int(value: str):
    if value.lower() in {"none", "all", "full"}:
        return None
    return int(value)


def _parse_smoothing_modes(value: str):
    valid_modes = set(DEFAULT_SMOOTHING_MODES)
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(modes) - valid_modes)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown smoothing mode(s): {', '.join(unknown)}")
    return modes


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _float_dict(metrics):
    return {key: float(value) for key, value in metrics.items()}


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return float(mean), 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(mean), float(np.sqrt(variance))


def _add_mean_std(rows, metric_name, values):
    mean, std = _mean_std(values)
    rows.append(
        {
            "metric": metric_name,
            "seeds": len(values),
            "mean": mean,
            "std": std,
        }
    )


def _add_grouped_mean_std(rows, metric_name, values_by_group):
    for group_key, values in sorted(values_by_group.items()):
        mean, std = _mean_std(values)
        row = {
            "metric": metric_name,
            "seeds": len(values),
            "mean": mean,
            "std": std,
        }
        if isinstance(group_key, tuple):
            for key, value in group_key:
                row[key] = value
        else:
            row["group"] = group_key
        rows.append(row)


def _summarize_multi_seed(payloads):
    rows = []
    clean_test_values = [payload["clean_metrics"]["test"] for payload in payloads]
    best_test_values = [payload["best_test_accuracy"] for payload in payloads]
    edge_drop_values = [
        payload["edge_drop_baseline"]["test"]
        for payload in payloads
        if payload.get("edge_drop_baseline") is not None
    ]
    sparse_values = [
        payload["sparse_flip_baseline"]["test"]
        for payload in payloads
        if payload.get("sparse_flip_baseline") is not None
    ]

    _add_mean_std(rows, "clean_test_accuracy", clean_test_values)
    _add_mean_std(rows, "best_test_accuracy", best_test_values)
    if edge_drop_values:
        _add_mean_std(rows, "edge_drop_smoothed_test_accuracy", edge_drop_values)
    if sparse_values:
        _add_mean_std(rows, "sparse_edge_flip_smoothed_test_accuracy", sparse_values)

    attack_by_budget = {}
    for payload in payloads:
        for row in payload.get("global_attack_results", []):
            attack_by_budget.setdefault((("budget", int(row["budget"])),), []).append(float(row["test_accuracy"]))
    _add_grouped_mean_std(rows, "prbcd_attacked_test_accuracy", attack_by_budget)

    symmetric_cert_by_flip = {}
    for payload in payloads:
        for row in payload.get("local_certificate_accuracy_subset_sweep", []):
            key = (("p_flip", float(row["p_flip"])),)
            symmetric_cert_by_flip.setdefault(key, []).append(float(row["positive_certified_accuracy"]))
    _add_grouped_mean_std(rows, "symmetric_positive_certified_accuracy", symmetric_cert_by_flip)

    sparse_cert_by_config = {}
    sparse_delete_cert_by_config = {}
    sparse_add_cert_by_config = {}
    for payload in payloads:
        for row in payload.get("sparse_asymmetric_certificate_accuracy_subset_sweep", []):
            key = (
                ("p_delete", float(row["p_delete"])),
                ("p_add", float(row["p_add"])),
                ("max_additions", int(row["max_additions"])),
            )
            sparse_cert_by_config.setdefault(key, []).append(float(row["positive_certified_accuracy"]))
            sparse_delete_cert_by_config.setdefault(key, []).append(float(row.get("positive_delete_certified_accuracy", 0.0)))
            sparse_add_cert_by_config.setdefault(key, []).append(float(row.get("positive_add_certified_accuracy", 0.0)))
    _add_grouped_mean_std(rows, "sparse_asymmetric_positive_certified_accuracy", sparse_cert_by_config)
    _add_grouped_mean_std(rows, "sparse_asymmetric_deletion_only_certified_accuracy", sparse_delete_cert_by_config)
    _add_grouped_mean_std(rows, "sparse_asymmetric_addition_only_certified_accuracy", sparse_add_cert_by_config)

    return rows



def _print_metrics(prefix, metrics):
    print(
        f"{prefix} | "
        f"Train {metrics['train']:.4f} | "
        f"Val {metrics['val']:.4f} | "
        f"Test {metrics['test']:.4f}"
    )


def _wilson_interval(successes: int, total: int, z: float = CONFIDENCE_Z):
    if total <= 0:
        return 0.0, 0.0

    phat = successes / total
    denominator = 1.0 + (z * z / total)
    center = phat + (z * z / (2.0 * total))
    margin = z * np.sqrt((phat * (1.0 - phat) + (z * z / (4.0 * total))) / total)
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return max(0.0, float(lower)), min(1.0, float(upper))


def _fraction_with_ci(successes: int, total: int):
    fraction = successes / total if total > 0 else 0.0
    lower, upper = _wilson_interval(successes, total)
    return float(fraction), lower, upper


def _select_certificate_nodes(data, max_nodes=None):
    test_nodes = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().tolist()
    if max_nodes is None:
        return [int(node_idx) for node_idx in test_nodes]
    if max_nodes >= len(test_nodes):
        return [int(node_idx) for node_idx in test_nodes]
    return sorted(int(node_idx) for node_idx in random.sample(test_nodes, max_nodes))


def _build_radius_curve(certificate_rows, radius_key, max_radius):
    total_nodes = len(certificate_rows)
    curve = []

    for radius in range(max_radius + 1):
        certified_count = sum(
            bool(row.get("is_correct", False)) and int(row.get(radius_key, 0) or 0) >= radius
            for row in certificate_rows
        )
        _, ci_lower, ci_upper = _fraction_with_ci(certified_count, total_nodes)
        curve.append(
            {
                "radius": int(radius),
                "certified_nodes": int(certified_count),
                "evaluated_nodes": int(total_nodes),
                "certified_accuracy": certified_count / total_nodes if total_nodes > 0 else 0.0,
                "certified_accuracy_ci_lower": ci_lower,
                "certified_accuracy_ci_upper": ci_upper,
            }
        )

    return curve


def _select_clean_certificate_focus(sparse_rows, symmetric_rows):
    candidates = []

    for row in sparse_rows:
        if not row.get("is_correct", False):
            continue
        candidates.append(
            {
                "node_idx": int(row["node_idx"]),
                "source": "sparse-asymmetric",
                "score": (
                    int(row.get("reported_asymmetric_delete_budget", 0))
                    + int(row.get("reported_asymmetric_add_budget", 0))
                    + int(row.get("reported_asymmetric_total_radius", 0)),
                    float(row.get("pA_lower", 0.0)),
                ),
            }
        )

    for row in symmetric_rows:
        if not row.get("is_correct", False):
            continue
        candidates.append(
            {
                "node_idx": int(row["node_idx"]),
                "source": "symmetric",
                "score": (
                    int(row.get("reported_certified_radius", 0)),
                    float(row.get("pA_lower", 0.0)),
                ),
            }
        )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["score"])



def _train_model(model, data, optimizer, epochs: int = 200):
    history = []
    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer)
        metrics, _ = evaluate(model, data)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss),
                "train": float(metrics["train"]),
                "val": float(metrics["val"]),
                "test": float(metrics["test"]),
            }
        )

        if metrics["val"] > best_val:
            best_val = metrics["val"]
            best_test = metrics["test"]

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Train {metrics['train']:.4f} | "
                f"Val {metrics['val']:.4f} | "
                f"Test {metrics['test']:.4f}"
            )

    print(f"\nBest Val:  {best_val:.4f}")
    print(f"Best Test: {best_test:.4f}")
    return history, best_val, best_test



def _select_focus_result(model, data, nettack_results):
    candidates = [result for result in nettack_results if result["success"]]
    if not candidates:
        candidates = nettack_results
    if not candidates:
        return None, None

    beta = 1.0 - FOCUS_SELECTION_P_FLIP
    best_bundle = None
    best_score = None

    for result in candidates:
        clean_certificate, _, _ = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=data.edge_index,
            node_idx=result["target_node"],
            num_samples=FOCUS_SELECTION_SAMPLES,
            mode="symmetric-edge-flip",
            p_delete=FOCUS_SELECTION_P_FLIP,
            certificate_beta=beta,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
        )
        attack_certificate, _, _ = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=result["attacked_edge_index"],
            node_idx=result["target_node"],
            num_samples=FOCUS_SELECTION_SAMPLES,
            mode="symmetric-edge-flip",
            p_delete=FOCUS_SELECTION_P_FLIP,
            certificate_beta=beta,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
        )

        score = (
            int(clean_certificate["is_correct"]),
            int(not attack_certificate["is_correct"]),
            int(result["success"]),
            clean_certificate["reported_certified_radius"] - attack_certificate["reported_certified_radius"],
            clean_certificate["pA_lower"] - attack_certificate["pA_lower"],
        )

        if best_score is None or score > best_score:
            best_score = score
            best_bundle = {
                "result": result,
                "clean_probe": clean_certificate,
                "attack_probe": attack_certificate,
            }

    return best_bundle["result"], best_bundle



def run_single_experiment(seed: int, results_dir: Path, dataset_name: str, smoothing_modes, epochs: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = Planetoid(
        root="data/Planetoid",
        name=dataset_name,
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=32,
        out_channels=dataset.num_classes,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    training_history, best_val, best_test = _train_model(model, data, optimizer, epochs=epochs)

    clean_metrics, _ = evaluate_with_edge_index(model, data, data.edge_index)
    print(f"\nClean test accuracy: {clean_metrics['test']:.4f}")

    edge_drop_baseline = None
    if "edge-drop" in smoothing_modes:
        print("\n=== Edge-drop smoothing baseline ===")
        edge_drop_baseline, _, _, _ = evaluate_smoothed(
            model=model,
            data=data,
            num_samples=500,
            batch_size=50,
            mode="edge-drop",
            p_delete=0.10,
        )
        _print_metrics("Edge-drop smoothing", edge_drop_baseline)

    sparse_baseline_config = SPARSE_FLIP_SWEEP[min(1, len(SPARSE_FLIP_SWEEP) - 1)] if SPARSE_FLIP_SWEEP else None
    sparse_flip_baseline = None
    if "sparse-edge-flip" in smoothing_modes and sparse_baseline_config is not None:
        print("\n=== Sparse edge-flip smoothing baseline ===")
        sparse_flip_baseline, _, _, _ = evaluate_smoothed(
            model=model,
            data=data,
            num_samples=500,
            batch_size=50,
            mode="sparse-edge-flip",
            p_delete=sparse_baseline_config["p_delete"],
            p_add=sparse_baseline_config["p_add"],
            max_additions=sparse_baseline_config["max_additions"],
        )
        _print_metrics("Sparse edge-flip smoothing", sparse_flip_baseline)

    print("\n=== Local test-node certified accuracy sweep ===")
    certificate_subset_nodes = _select_certificate_nodes(data, max_nodes=CERTIFIED_ACCURACY_NODE_COUNT)
    print(
        f"Evaluating {len(certificate_subset_nodes)} test nodes with "
        f"{CERTIFIED_ACCURACY_SAMPLES} smoothing samples per node"
    )
    symmetric_certificate_rows = []
    sparse_asymmetric_certificate_rows = []
    sparse_asymmetric_curve_rows = []
    certified_accuracy_curve_rows = []
    certified_accuracy_curves = []
    sparse_asymmetric_curves = []
    certificate_subset_rows = []
    sparse_asymmetric_subset_rows = []
    clean_certificate_focus_rows = []
    clean_certificate_focus_summary = None

    if "symmetric-cert" in smoothing_modes:
        for p_flip in CERTIFIED_ACCURACY_SWEEP:
            per_node_certificates = []
            print(f"Symmetric certificate p_flip={p_flip:.2f}: starting {len(certificate_subset_nodes)} nodes")
            for node_position, node_idx in enumerate(certificate_subset_nodes, start=1):
                if node_position == 1 or node_position % 10 == 0 or node_position == len(certificate_subset_nodes):
                    print(
                        f"  symmetric p_flip={p_flip:.2f}: "
                        f"node {node_position}/{len(certificate_subset_nodes)}"
                    )
                certificate, _, _ = evaluate_smoothed_node_with_edge_index(
                    model=model,
                    data=data,
                    edge_index=data.edge_index,
                    node_idx=node_idx,
                    num_samples=CERTIFIED_ACCURACY_SAMPLES,
                    mode="symmetric-edge-flip",
                    p_delete=p_flip,
                    certificate_beta=1.0 - p_flip,
                    certificate_max_radius=CERTIFICATE_MAX_RADIUS,
                )
                certificate["node_idx"] = int(node_idx)
                certificate["p_flip"] = float(p_flip)
                per_node_certificates.append(certificate)
                certificate_subset_rows.append(
                    {
                        "p_flip": float(p_flip),
                        "node_idx": int(node_idx),
                        "predicted_class": int(certificate["predicted_class"]),
                        "true_label": int(certificate["true_label"]),
                        "is_correct": bool(certificate["is_correct"]),
                        "certified_radius": int(certificate["certified_radius"]),
                        "reported_certified_radius": int(certificate["reported_certified_radius"]),
                        "pA_lower": float(certificate["pA_lower"]),
                    }
                )

            total_nodes = len(per_node_certificates)
            correct_nodes = sum(certificate["is_correct"] for certificate in per_node_certificates)
            reported_radii = [
                certificate["reported_certified_radius"]
                for certificate in per_node_certificates
                if certificate["is_correct"]
            ]
            positive_certified_nodes = sum(radius > 0 for radius in reported_radii)
            correct_fraction, correct_ci_lower, correct_ci_upper = _fraction_with_ci(correct_nodes, total_nodes)
            positive_certified_accuracy, positive_ci_lower, positive_ci_upper = _fraction_with_ci(
                positive_certified_nodes,
                total_nodes,
            )
            certified_fraction_on_correct, certified_on_correct_ci_lower, certified_on_correct_ci_upper = _fraction_with_ci(
                positive_certified_nodes,
                correct_nodes,
            )
            certified_accuracy_curve = build_certified_accuracy_curve(
                per_node_certificates,
                max_radius=CERTIFICATE_MAX_RADIUS,
            )

            row = {
                "p_flip": float(p_flip),
                "evaluated_nodes": int(total_nodes),
                "correct_nodes": int(correct_nodes),
                "correct_fraction": correct_fraction,
                "correct_fraction_ci_lower": correct_ci_lower,
                "correct_fraction_ci_upper": correct_ci_upper,
                "positive_certified_nodes": int(positive_certified_nodes),
                "positive_certified_accuracy": positive_certified_accuracy,
                "positive_certified_accuracy_ci_lower": positive_ci_lower,
                "positive_certified_accuracy_ci_upper": positive_ci_upper,
                "certified_fraction_on_correct": certified_fraction_on_correct,
                "certified_fraction_on_correct_ci_lower": certified_on_correct_ci_lower,
                "certified_fraction_on_correct_ci_upper": certified_on_correct_ci_upper,
                "mean_certified_radius_on_correct": sum(reported_radii) / len(reported_radii) if reported_radii else 0.0,
            }
            symmetric_certificate_rows.append(row)

            curve_rows = []
            for point in certified_accuracy_curve:
                certified_count = sum(
                    bool(certificate["is_correct"]) and int(certificate["certified_radius"] or 0) >= int(point["radius"])
                    for certificate in per_node_certificates
                )
                _, curve_ci_lower, curve_ci_upper = _fraction_with_ci(certified_count, total_nodes)
                curve_row = {
                    "p_flip": float(p_flip),
                    "radius": int(point["radius"]),
                    "certified_nodes": int(certified_count),
                    "evaluated_nodes": int(total_nodes),
                    "certified_accuracy": float(point["certified_accuracy"]),
                    "certified_accuracy_ci_lower": curve_ci_lower,
                    "certified_accuracy_ci_upper": curve_ci_upper,
                }
                certified_accuracy_curve_rows.append(curve_row)
                curve_rows.append({"radius": curve_row["radius"], "certified_accuracy": curve_row["certified_accuracy"]})

            certified_accuracy_curves.append(
                {
                    "label": f"p_flip={p_flip:.2f}",
                    "rows": curve_rows,
                }
            )

            print(
                f"p_flip={p_flip:.2f} | "
                f"Test accuracy {row['correct_fraction']:.4f} "
                f"[{row['correct_fraction_ci_lower']:.4f}, {row['correct_fraction_ci_upper']:.4f}] | "
                f"Certified acc @ r>=1 {row['positive_certified_accuracy']:.4f} "
                f"[{row['positive_certified_accuracy_ci_lower']:.4f}, {row['positive_certified_accuracy_ci_upper']:.4f}] | "
                f"Mean radius on correct {row['mean_certified_radius_on_correct']:.3f}"
            )

    if "sparse-asymmetric-cert" in smoothing_modes:
        print("\n=== Local sparse asymmetric certified accuracy test-node sweep ===")
    for config in SPARSE_FLIP_SWEEP if "sparse-asymmetric-cert" in smoothing_modes else []:
        p_delete = config["p_delete"]
        p_add = config["p_add"]
        per_node_certificates = []

        print(
            f"Sparse asymmetric certificate p_delete={p_delete:.3f}, p_add={p_add:.6f}: "
            f"starting {len(certificate_subset_nodes)} nodes"
        )
        for node_position, node_idx in enumerate(certificate_subset_nodes, start=1):
            if node_position == 1 or node_position % 10 == 0 or node_position == len(certificate_subset_nodes):
                print(
                    f"  sparse p_delete={p_delete:.3f}, p_add={p_add:.6f}: "
                    f"node {node_position}/{len(certificate_subset_nodes)}"
                )
            certificate, _, _ = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=data.edge_index,
                node_idx=node_idx,
                num_samples=CERTIFIED_ACCURACY_SAMPLES,
                mode="sparse-edge-flip",
                p_delete=p_delete,
                p_add=p_add,
                max_additions=config["max_additions"],
                certificate_p_delete=p_delete,
                certificate_p_add=p_add,
                certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
            )
            asymmetric = certificate["asymmetric_certificate"]
            total_radius = int(asymmetric["total_radius"])
            row_certificate = {
                **certificate,
                "certified_radius": int(certificate["reported_asymmetric_total_radius"]),
                "delete_radius": int(certificate["reported_asymmetric_delete_budget"]),
                "add_radius": int(certificate["reported_asymmetric_add_budget"]),
            }
            per_node_certificates.append(row_certificate)
            sparse_asymmetric_subset_rows.append(
                {
                    "p_delete": float(p_delete),
                    "p_add": float(p_add),
                    "max_additions": int(config["max_additions"]),
                    "node_idx": int(node_idx),
                    "predicted_class": int(certificate["predicted_class"]),
                    "true_label": int(certificate["true_label"]),
                    "is_correct": bool(certificate["is_correct"]),
                    "asymmetric_total_radius": total_radius,
                    "reported_asymmetric_total_radius": int(certificate["reported_asymmetric_total_radius"]),
                    "max_delete_budget": int(asymmetric["max_delete_budget"]),
                    "max_add_budget": int(asymmetric["max_add_budget"]),
                    "reported_asymmetric_delete_budget": int(certificate["reported_asymmetric_delete_budget"]),
                    "reported_asymmetric_add_budget": int(certificate["reported_asymmetric_add_budget"]),
                    "pA_lower": float(certificate["pA_lower"]),
                }
            )

        total_nodes = len(per_node_certificates)
        correct_nodes = sum(certificate["is_correct"] for certificate in per_node_certificates)
        reported_radii = [
            certificate["reported_asymmetric_total_radius"]
            for certificate in per_node_certificates
            if certificate["is_correct"]
        ]
        reported_delete_radii = [
            certificate["reported_asymmetric_delete_budget"]
            for certificate in per_node_certificates
            if certificate["is_correct"]
        ]
        reported_add_radii = [
            certificate["reported_asymmetric_add_budget"]
            for certificate in per_node_certificates
            if certificate["is_correct"]
        ]
        positive_certified_nodes = sum(radius > 0 for radius in reported_radii)
        positive_delete_certified_nodes = sum(radius > 0 for radius in reported_delete_radii)
        positive_add_certified_nodes = sum(radius > 0 for radius in reported_add_radii)
        correct_fraction, correct_ci_lower, correct_ci_upper = _fraction_with_ci(correct_nodes, total_nodes)
        positive_certified_accuracy, positive_ci_lower, positive_ci_upper = _fraction_with_ci(
            positive_certified_nodes,
            total_nodes,
        )
        positive_delete_accuracy, positive_delete_ci_lower, positive_delete_ci_upper = _fraction_with_ci(
            positive_delete_certified_nodes,
            total_nodes,
        )
        positive_add_accuracy, positive_add_ci_lower, positive_add_ci_upper = _fraction_with_ci(
            positive_add_certified_nodes,
            total_nodes,
        )
        certified_fraction_on_correct, certified_on_correct_ci_lower, certified_on_correct_ci_upper = _fraction_with_ci(
            positive_certified_nodes,
            correct_nodes,
        )
        certified_accuracy_curve = build_certified_accuracy_curve(
            per_node_certificates,
            max_radius=ASYMMETRIC_CERTIFICATE_MAX_DELETE + ASYMMETRIC_CERTIFICATE_MAX_ADD,
        )
        row = {
            "p_delete": float(p_delete),
            "p_add": float(p_add),
            "max_additions": int(config["max_additions"]),
            "evaluated_nodes": int(total_nodes),
            "correct_nodes": int(correct_nodes),
            "correct_fraction": correct_fraction,
            "correct_fraction_ci_lower": correct_ci_lower,
            "correct_fraction_ci_upper": correct_ci_upper,
            "positive_certified_nodes": int(positive_certified_nodes),
            "positive_certified_accuracy": positive_certified_accuracy,
            "positive_certified_accuracy_ci_lower": positive_ci_lower,
            "positive_certified_accuracy_ci_upper": positive_ci_upper,
            "positive_delete_certified_nodes": int(positive_delete_certified_nodes),
            "positive_delete_certified_accuracy": positive_delete_accuracy,
            "positive_delete_certified_accuracy_ci_lower": positive_delete_ci_lower,
            "positive_delete_certified_accuracy_ci_upper": positive_delete_ci_upper,
            "positive_add_certified_nodes": int(positive_add_certified_nodes),
            "positive_add_certified_accuracy": positive_add_accuracy,
            "positive_add_certified_accuracy_ci_lower": positive_add_ci_lower,
            "positive_add_certified_accuracy_ci_upper": positive_add_ci_upper,
            "certified_fraction_on_correct": certified_fraction_on_correct,
            "certified_fraction_on_correct_ci_lower": certified_on_correct_ci_lower,
            "certified_fraction_on_correct_ci_upper": certified_on_correct_ci_upper,
            "mean_certified_radius_on_correct": sum(reported_radii) / len(reported_radii) if reported_radii else 0.0,
            "mean_delete_radius_on_correct": sum(reported_delete_radii) / len(reported_delete_radii)
            if reported_delete_radii
            else 0.0,
            "mean_add_radius_on_correct": sum(reported_add_radii) / len(reported_add_radii) if reported_add_radii else 0.0,
        }
        sparse_asymmetric_certificate_rows.append(row)

        curve_rows = []
        for point in certified_accuracy_curve:
            certified_count = sum(
                bool(certificate["is_correct"]) and int(certificate["certified_radius"] or 0) >= int(point["radius"])
                for certificate in per_node_certificates
            )
            _, curve_ci_lower, curve_ci_upper = _fraction_with_ci(certified_count, total_nodes)
            curve_row = {
                "p_delete": float(p_delete),
                "p_add": float(p_add),
                "radius_type": "total",
                "radius": int(point["radius"]),
                "certified_nodes": int(certified_count),
                "evaluated_nodes": int(total_nodes),
                "certified_accuracy": float(point["certified_accuracy"]),
                "certified_accuracy_ci_lower": curve_ci_lower,
                "certified_accuracy_ci_upper": curve_ci_upper,
            }
            sparse_asymmetric_curve_rows.append(curve_row)
            curve_rows.append({"radius": curve_row["radius"], "certified_accuracy": curve_row["certified_accuracy"]})

        sparse_asymmetric_curves.append(
            {
                "label": f"p_del={p_delete:.2f}, p_add={p_add:.0e}",
                "rows": curve_rows,
            }
        )

        for point in _build_radius_curve(
            per_node_certificates,
            radius_key="delete_radius",
            max_radius=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
        ):
            sparse_asymmetric_curve_rows.append(
                {
                    "p_delete": float(p_delete),
                    "p_add": float(p_add),
                    "radius_type": "deletion_only",
                    **point,
                }
            )

        for point in _build_radius_curve(
            per_node_certificates,
            radius_key="add_radius",
            max_radius=ASYMMETRIC_CERTIFICATE_MAX_ADD,
        ):
            sparse_asymmetric_curve_rows.append(
                {
                    "p_delete": float(p_delete),
                    "p_add": float(p_add),
                    "radius_type": "addition_only",
                    **point,
                }
            )

        print(
            f"p_delete={p_delete:.3f}, p_add={p_add:.6f} | "
            f"Test accuracy {row['correct_fraction']:.4f} "
            f"[{row['correct_fraction_ci_lower']:.4f}, {row['correct_fraction_ci_upper']:.4f}] | "
            f"Certified acc @ r>=1 {row['positive_certified_accuracy']:.4f} "
            f"[{row['positive_certified_accuracy_ci_lower']:.4f}, {row['positive_certified_accuracy_ci_upper']:.4f}] | "
            f"Del-only @ r>=1 {row['positive_delete_certified_accuracy']:.4f} | "
            f"Add-only @ r>=1 {row['positive_add_certified_accuracy']:.4f} | "
            f"Mean total radius on correct {row['mean_certified_radius_on_correct']:.3f}"
        )

    clean_certificate_focus = _select_clean_certificate_focus(
        sparse_rows=sparse_asymmetric_subset_rows,
        symmetric_rows=certificate_subset_rows,
    )
    if clean_certificate_focus is not None:
        clean_focus_node = int(clean_certificate_focus["node_idx"])
        clean_certificate_focus_summary = {
            "target_node": clean_focus_node,
            "selection_source": clean_certificate_focus["source"],
            "selection_score": list(clean_certificate_focus["score"]),
            "true_label": int(data.y[clean_focus_node].item()),
        }
        print(
            f"\nSelected clean certification node {clean_focus_node} "
            f"from {clean_certificate_focus['source']} candidates"
        )

        if "symmetric-cert" in smoothing_modes:
            for p_flip in LOCAL_CERTIFICATE_SWEEP:
                beta = 1.0 - p_flip
                certificate, _, pred = evaluate_smoothed_node_with_edge_index(
                    model=model,
                    data=data,
                    edge_index=data.edge_index,
                    node_idx=clean_focus_node,
                    num_samples=LOCAL_CERTIFICATE_SAMPLES,
                    mode="symmetric-edge-flip",
                    p_delete=p_flip,
                    certificate_beta=beta,
                    certificate_max_radius=CERTIFICATE_MAX_RADIUS,
                )
                clean_certificate_focus_rows.append(
                    {
                        "mode": "symmetric-edge-flip",
                        "p_flip": float(p_flip),
                        "p_delete": "",
                        "p_add": "",
                        "max_additions": "",
                        "predicted_class": int(pred),
                        "true_label": int(data.y[clean_focus_node].item()),
                        "is_correct": bool(certificate["is_correct"]),
                        "reported_total_radius": int(certificate["reported_certified_radius"]),
                        "reported_delete_budget": "",
                        "reported_add_budget": "",
                        "pA_lower": float(certificate["pA_lower"]),
                    }
                )

        if "sparse-asymmetric-cert" in smoothing_modes:
            for config in SPARSE_FLIP_SWEEP:
                p_delete = config["p_delete"]
                p_add = config["p_add"]
                certificate, _, pred = evaluate_smoothed_node_with_edge_index(
                    model=model,
                    data=data,
                    edge_index=data.edge_index,
                    node_idx=clean_focus_node,
                    num_samples=LOCAL_CERTIFICATE_SAMPLES,
                    mode="sparse-edge-flip",
                    p_delete=p_delete,
                    p_add=p_add,
                    max_additions=config["max_additions"],
                    certificate_p_delete=p_delete,
                    certificate_p_add=p_add,
                    certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                    certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
                )
                clean_certificate_focus_rows.append(
                    {
                        "mode": "sparse-edge-flip",
                        "p_flip": "",
                        "p_delete": float(p_delete),
                        "p_add": float(p_add),
                        "max_additions": int(config["max_additions"]),
                        "predicted_class": int(pred),
                        "true_label": int(data.y[clean_focus_node].item()),
                        "is_correct": bool(certificate["is_correct"]),
                        "reported_total_radius": int(certificate["reported_asymmetric_total_radius"]),
                        "reported_delete_budget": int(certificate["reported_asymmetric_delete_budget"]),
                        "reported_add_budget": int(certificate["reported_asymmetric_add_budget"]),
                        "pA_lower": float(certificate["pA_lower"]),
                    }
                )

    global_attack_rows = []
    print("\n=== Global attack evaluation ===")
    for budget in GLOBAL_ATTACK_BUDGETS:
        attacked_edge_index, flipped_edges = run_prbcd_attack(model, data, budget, device)
        attacked_metrics, _ = evaluate_with_edge_index(model, data, attacked_edge_index)
        num_flips = flipped_edges.size(1) if flipped_edges.numel() > 0 else 0

        global_attack_rows.append(
            {
                "budget": budget,
                "test_accuracy": float(attacked_metrics["test"]),
                "flipped_edges": int(num_flips),
            }
        )

        print(
            f"Budget {budget:>3d} | "
            f"Attacked test accuracy: {attacked_metrics['test']:.4f} | "
            f"Flipped edges: {num_flips}"
        )

    nettack_logs = []
    nettack_results = []

    if NETTACK_TARGET_COUNT > 0:
        print("\n=== Nettack targeted demo ===")
        surrogate, adj, features, labels = train_deeprobust_surrogate(data, device=device)
        candidate_nodes = choose_correct_test_nodes(model, data, k=NETTACK_TARGET_COUNT)

        for target_node in candidate_nodes:
            result = run_nettack_on_node(
                target_model=model,
                data=data,
                surrogate=surrogate,
                adj=adj,
                features=features,
                labels=labels,
                target_node=target_node,
                n_perturbations=3,
                device=device,
            )

            nettack_results.append(result)
            nettack_logs.append(
                {
                    "target_node": int(result["target_node"]),
                    "true_label": int(result["true_label"]),
                    "clean_pred": int(result["clean_pred"]),
                    "attacked_pred": int(result["attacked_pred"]),
                    "success": bool(result["success"]),
                    "perturbations": str(result["perturbations"]),
                }
            )

            print(
                f"Node {result['target_node']} | "
                f"True {result['true_label']} | "
                f"Clean pred {result['clean_pred']} | "
                f"Attacked pred {result['attacked_pred']} | "
                f"Success: {result['success']}"
            )
            print("Structure perturbations:", result["perturbations"])

    focus_result, focus_probe = _select_focus_result(model, data, nettack_results)

    edge_drop_rows = []
    sparse_flip_rows = []
    certificate_rows = []
    asymmetric_certificate_rows = []
    selected_focus_summary = None

    if focus_result is not None:
        node_idx = focus_result["target_node"]
        attacked_edge_index = focus_result["attacked_edge_index"]
        selected_focus_summary = {
            "target_node": int(node_idx),
            "true_label": int(focus_result["true_label"]),
            "clean_probe_correct": bool(focus_probe["clean_probe"]["is_correct"]),
            "attack_probe_correct": bool(focus_probe["attack_probe"]["is_correct"]),
            "clean_probe_radius": int(focus_probe["clean_probe"]["reported_certified_radius"]),
            "attack_probe_radius": int(focus_probe["attack_probe"]["reported_certified_radius"]),
        }

        print(
            f"\nSelected Nettack target node {node_idx} for the attack case study | "
            f"Clean probe correct: {focus_probe['clean_probe']['is_correct']} | "
            f"Attack probe correct: {focus_probe['attack_probe']['is_correct']}"
        )

        if "edge-drop" in smoothing_modes:
            print("\n=== Edge-drop smoothing sweep on clean vs Nettack graph ===")
        for p_delete in EDGE_DROP_SWEEP if "edge-drop" in smoothing_modes else []:
            clean_metrics_s, clean_votes, clean_pred, _ = evaluate_smoothed_with_edge_index(
                model=model,
                data=data,
                edge_index=data.edge_index,
                num_samples=500,
                batch_size=50,
                mode="edge-drop",
                p_delete=p_delete,
            )
            attack_metrics_s, attack_votes, attack_pred, _ = evaluate_smoothed_with_edge_index(
                model=model,
                data=data,
                edge_index=attacked_edge_index,
                num_samples=500,
                batch_size=50,
                mode="edge-drop",
                p_delete=p_delete,
            )

            clean_cert = certify_node_from_votes(clean_votes, node_idx=node_idx)
            attack_cert = certify_node_from_votes(attack_votes, node_idx=node_idx)
            true_label = int(data.y[node_idx].item())

            row = {
                "p_delete": float(p_delete),
                "clean_test_accuracy": float(clean_metrics_s["test"]),
                "attacked_test_accuracy": float(attack_metrics_s["test"]),
                "clean_margin": float(clean_cert["margin"]),
                "attacked_margin": float(attack_cert["margin"]),
                "clean_pA_lower": float(clean_cert["pA_lower"]),
                "attacked_pA_lower": float(attack_cert["pA_lower"]),
                "clean_pred": int(clean_pred[node_idx].item()),
                "attacked_pred": int(attack_pred[node_idx].item()),
                "clean_is_correct": int(clean_pred[node_idx].item() == true_label),
                "attacked_is_correct": int(attack_pred[node_idx].item() == true_label),
                "true_label": true_label,
            }
            edge_drop_rows.append(row)

            print(
                f"p_delete={p_delete:.2f} | "
                f"Clean acc {clean_metrics_s['test']:.4f} | "
                f"Attack acc {attack_metrics_s['test']:.4f} | "
                f"Clean margin {clean_cert['margin']:.3f} | "
                f"Attack margin {attack_cert['margin']:.3f}"
            )

        if "sparse-edge-flip" in smoothing_modes:
            print("\n=== Sparse edge-flip smoothing sweep ===")
        for config in SPARSE_FLIP_SWEEP if "sparse-edge-flip" in smoothing_modes else []:
            p_delete = config["p_delete"]
            p_add = config["p_add"]
            max_additions = config["max_additions"]

            clean_metrics_s, clean_votes, clean_pred, _ = evaluate_smoothed_with_edge_index(
                model=model,
                data=data,
                edge_index=data.edge_index,
                num_samples=500,
                batch_size=50,
                mode="sparse-edge-flip",
                p_delete=p_delete,
                p_add=p_add,
                max_additions=max_additions,
            )
            attack_metrics_s, attack_votes, attack_pred, _ = evaluate_smoothed_with_edge_index(
                model=model,
                data=data,
                edge_index=attacked_edge_index,
                num_samples=500,
                batch_size=50,
                mode="sparse-edge-flip",
                p_delete=p_delete,
                p_add=p_add,
                max_additions=max_additions,
            )

            clean_cert = certify_node_from_votes(clean_votes, node_idx=node_idx)
            attack_cert = certify_node_from_votes(attack_votes, node_idx=node_idx)
            true_label = int(data.y[node_idx].item())

            row = {
                "p_delete": float(p_delete),
                "p_add": float(p_add),
                "max_additions": int(max_additions),
                "clean_test_accuracy": float(clean_metrics_s["test"]),
                "attacked_test_accuracy": float(attack_metrics_s["test"]),
                "clean_margin": float(clean_cert["margin"]),
                "attacked_margin": float(attack_cert["margin"]),
                "clean_pred": int(clean_pred[node_idx].item()),
                "attacked_pred": int(attack_pred[node_idx].item()),
                "clean_is_correct": int(clean_pred[node_idx].item() == true_label),
                "attacked_is_correct": int(attack_pred[node_idx].item() == true_label),
                "true_label": true_label,
            }
            sparse_flip_rows.append(row)

            print(
                f"p_delete={p_delete:.3f}, p_add={p_add:.6f} | "
                f"Clean acc {clean_metrics_s['test']:.4f} | "
                f"Attack acc {attack_metrics_s['test']:.4f}"
            )

        if "symmetric-cert" in smoothing_modes:
            print("\n=== Target-node symmetric edge-flip certificate sweep ===")
        for p_flip in LOCAL_CERTIFICATE_SWEEP if "symmetric-cert" in smoothing_modes else []:
            beta = 1.0 - p_flip
            clean_certificate, _, clean_pred = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=data.edge_index,
                node_idx=node_idx,
                num_samples=LOCAL_CERTIFICATE_SAMPLES,
                mode="symmetric-edge-flip",
                p_delete=p_flip,
                certificate_beta=beta,
                certificate_max_radius=CERTIFICATE_MAX_RADIUS,
            )
            attack_certificate, _, attack_pred = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=attacked_edge_index,
                node_idx=node_idx,
                num_samples=LOCAL_CERTIFICATE_SAMPLES,
                mode="symmetric-edge-flip",
                p_delete=p_flip,
                certificate_beta=beta,
                certificate_max_radius=CERTIFICATE_MAX_RADIUS,
            )

            certificate_rows.append(
                {
                    "p_flip": float(p_flip),
                    "beta": float(beta),
                    "clean_certified_radius": int(clean_certificate["certified_radius"]),
                    "attacked_certified_radius": int(attack_certificate["certified_radius"]),
                    "clean_reported_certified_radius": int(clean_certificate["reported_certified_radius"]),
                    "attacked_reported_certified_radius": int(attack_certificate["reported_certified_radius"]),
                    "clean_runner_up_radius": int(clean_certificate["runner_up_certified_radius"]),
                    "attacked_runner_up_radius": int(attack_certificate["runner_up_certified_radius"]),
                    "clean_pA_lower": float(clean_certificate["pA_lower"]),
                    "attacked_pA_lower": float(attack_certificate["pA_lower"]),
                    "clean_pred": int(clean_pred),
                    "attacked_pred": int(attack_pred),
                    "clean_is_correct": bool(clean_certificate["is_correct"]),
                    "attacked_is_correct": bool(attack_certificate["is_correct"]),
                    "true_label": int(data.y[node_idx].item()),
                }
            )

            print(
                f"p_flip={p_flip:.2f} | "
                f"Clean reportable radius {clean_certificate['reported_certified_radius']} | "
                f"Attack reportable radius {attack_certificate['reported_certified_radius']} | "
                f"Clean pA_lower {clean_certificate['pA_lower']:.3f} | "
                f"Attack pA_lower {attack_certificate['pA_lower']:.3f}"
            )

        if "sparse-asymmetric-cert" in smoothing_modes:
            print("\n=== Target-node asymmetric sparse edge-flip certificate sweep ===")
        for config in SPARSE_FLIP_SWEEP if "sparse-asymmetric-cert" in smoothing_modes else []:
            p_delete = config["p_delete"]
            p_add = config["p_add"]
            clean_certificate, _, clean_pred = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=data.edge_index,
                node_idx=node_idx,
                num_samples=LOCAL_CERTIFICATE_SAMPLES,
                mode="sparse-edge-flip",
                p_delete=p_delete,
                p_add=p_add,
                max_additions=config["max_additions"],
                certificate_p_delete=p_delete,
                certificate_p_add=p_add,
                certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
            )
            attack_certificate, _, attack_pred = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=attacked_edge_index,
                node_idx=node_idx,
                num_samples=LOCAL_CERTIFICATE_SAMPLES,
                mode="sparse-edge-flip",
                p_delete=p_delete,
                p_add=p_add,
                max_additions=config["max_additions"],
                certificate_p_delete=p_delete,
                certificate_p_add=p_add,
                certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
            )
            clean_asymmetric = clean_certificate["asymmetric_certificate"]
            attack_asymmetric = attack_certificate["asymmetric_certificate"]

            asymmetric_certificate_rows.append(
                {
                    "p_delete": float(p_delete),
                    "p_add": float(p_add),
                    "max_additions": int(config["max_additions"]),
                    "clean_total_radius": int(clean_asymmetric["total_radius"]),
                    "attacked_total_radius": int(attack_asymmetric["total_radius"]),
                    "clean_reported_total_radius": int(clean_certificate["reported_asymmetric_total_radius"]),
                    "attacked_reported_total_radius": int(attack_certificate["reported_asymmetric_total_radius"]),
                    "clean_delete_budget": int(clean_asymmetric["max_delete_budget"]),
                    "attacked_delete_budget": int(attack_asymmetric["max_delete_budget"]),
                    "clean_add_budget": int(clean_asymmetric["max_add_budget"]),
                    "attacked_add_budget": int(attack_asymmetric["max_add_budget"]),
                    "clean_pA_lower": float(clean_certificate["pA_lower"]),
                    "attacked_pA_lower": float(attack_certificate["pA_lower"]),
                    "clean_pred": int(clean_pred),
                    "attacked_pred": int(attack_pred),
                    "clean_is_correct": bool(clean_certificate["is_correct"]),
                    "attacked_is_correct": bool(attack_certificate["is_correct"]),
                    "true_label": int(data.y[node_idx].item()),
                }
            )

            print(
                f"p_delete={p_delete:.3f}, p_add={p_add:.6f} | "
                f"Clean reportable total radius {clean_certificate['reported_asymmetric_total_radius']} | "
                f"Attack reportable total radius {attack_certificate['reported_asymmetric_total_radius']} | "
                f"Clean del/add ({clean_asymmetric['max_delete_budget']}, {clean_asymmetric['max_add_budget']}) | "
                f"Attack del/add ({attack_asymmetric['max_delete_budget']}, {attack_asymmetric['max_add_budget']})"
            )

    results_payload = {
        "seed": int(seed),
        "dataset": dataset.name,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1) // 2),
        "best_validation_accuracy": float(best_val),
        "best_test_accuracy": float(best_test),
        "clean_metrics": _float_dict(clean_metrics),
        "certified_accuracy_evaluation": {
            "node_source": "test_mask",
            "node_selection": "seeded random subset" if CERTIFIED_ACCURACY_NODE_COUNT is not None else "full test mask",
            "max_nodes": CERTIFIED_ACCURACY_NODE_COUNT,
            "evaluated_nodes": int(len(certificate_subset_nodes)),
            "samples_per_node": int(CERTIFIED_ACCURACY_SAMPLES),
            "confidence_interval": "Wilson 95%",
            "confidence_z": float(CONFIDENCE_Z),
        },
        "edge_drop_baseline": _float_dict(edge_drop_baseline) if edge_drop_baseline is not None else None,
        "sparse_flip_baseline": (
            {
                **_float_dict(sparse_flip_baseline),
                **{
                    "p_delete": float(sparse_baseline_config["p_delete"]),
                    "p_add": float(sparse_baseline_config["p_add"]),
                    "max_additions": int(sparse_baseline_config["max_additions"]),
                },
            }
            if sparse_flip_baseline is not None
            else None
        ),
        "local_certificate_accuracy_subset_nodes": certificate_subset_nodes,
        "local_certificate_accuracy_subset_sweep": symmetric_certificate_rows,
        "local_certified_accuracy_curve": certified_accuracy_curve_rows,
        "local_certificate_subset_rows": certificate_subset_rows,
        "clean_certificate_focus": clean_certificate_focus_summary,
        "clean_certificate_focus_sweep": clean_certificate_focus_rows,
        "sparse_asymmetric_certificate_accuracy_subset_sweep": sparse_asymmetric_certificate_rows,
        "sparse_asymmetric_certified_accuracy_curve": sparse_asymmetric_curve_rows,
        "sparse_asymmetric_certificate_subset_rows": sparse_asymmetric_subset_rows,
        "global_attack_results": global_attack_rows,
        "nettack_examples": nettack_logs,
        "selected_focus_node": selected_focus_summary,
        "edge_drop_sweep": edge_drop_rows,
        "sparse_edge_flip_sweep": sparse_flip_rows,
        "local_certificate_sweep": certificate_rows,
        "local_asymmetric_certificate_sweep": asymmetric_certificate_rows,
        "artifacts_dir": str(results_dir),
    }

    save_csv_rows(results_dir / "training_history.csv", training_history)
    save_csv_rows(results_dir / "symmetric_certificate_sweep.csv", symmetric_certificate_rows)
    save_csv_rows(results_dir / "certified_accuracy_curve.csv", certified_accuracy_curve_rows)
    save_csv_rows(results_dir / "certificate_subset_rows.csv", certificate_subset_rows)
    save_csv_rows(results_dir / "clean_certificate_focus_sweep.csv", clean_certificate_focus_rows)
    save_csv_rows(results_dir / "sparse_asymmetric_certificate_sweep.csv", sparse_asymmetric_certificate_rows)
    save_csv_rows(results_dir / "sparse_asymmetric_certified_accuracy_curve.csv", sparse_asymmetric_curve_rows)
    save_csv_rows(results_dir / "sparse_asymmetric_certificate_subset_rows.csv", sparse_asymmetric_subset_rows)
    save_csv_rows(results_dir / "attack_budget_accuracy.csv", global_attack_rows)
    save_csv_rows(results_dir / "edge_drop_sweep.csv", edge_drop_rows)
    save_csv_rows(results_dir / "sparse_edge_flip_sweep.csv", sparse_flip_rows)
    save_csv_rows(results_dir / "local_certificate_sweep.csv", certificate_rows)
    save_csv_rows(results_dir / "local_asymmetric_certificate_sweep.csv", asymmetric_certificate_rows)
    save_json_report(results_dir / "preliminary_results.json", results_payload)

    plot_attack_budget_accuracy(
        output_path=results_dir / "attack_budget_accuracy.png",
        rows=global_attack_rows,
        clean_accuracy=clean_metrics["test"],
    )
    plot_certified_accuracy_curves(
        output_path=results_dir / "certified_accuracy_curve.png",
        curves=certified_accuracy_curves,
    )
    plot_certified_accuracy_curves(
        output_path=results_dir / "sparse_asymmetric_certified_accuracy_curve.png",
        curves=sparse_asymmetric_curves,
    )
    plot_dual_accuracy_curve(
        output_path=results_dir / "edge_drop_tradeoff.png",
        rows=edge_drop_rows,
        x_key="p_delete",
        clean_key="clean_test_accuracy",
        attacked_key="attacked_test_accuracy",
        title="Edge-drop smoothing tradeoff",
        x_label="Edge deletion probability",
    )
    plot_dual_accuracy_curve(
        output_path=results_dir / "sparse_flip_tradeoff.png",
        rows=sparse_flip_rows,
        x_key="p_delete",
        clean_key="clean_test_accuracy",
        attacked_key="attacked_test_accuracy",
        title="Sparse edge-flip smoothing tradeoff",
        x_label="Edge deletion probability",
    )
    plot_certificate_radius(
        output_path=results_dir / "local_certificate_radius.png",
        rows=certificate_rows,
    )

    print(f"\nSaved result artifacts to: {results_dir.resolve()}")
    return results_payload


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run certified robustness experiments for GNNs.")
    parser.add_argument("--dataset", default="Cora", help="Planetoid dataset name, e.g. Cora, CiteSeer, PubMed.")
    parser.add_argument("--seed", type=int, default=None, help="Single random seed. Ignored when --seeds is set.")
    parser.add_argument("--seeds", type=_parse_seed_list, default=None, help="Comma-separated seeds, e.g. 42,43,44.")
    parser.add_argument("--output-dir", default="results", help="Directory where artifacts are written.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per seed.")
    parser.add_argument(
        "--attack-budgets",
        type=_parse_int_list,
        default=GLOBAL_ATTACK_BUDGETS,
        help="Comma-separated PRBCD budgets. Use an empty string to skip PRBCD.",
    )
    parser.add_argument(
        "--edge-drop-sweep",
        type=_parse_float_list,
        default=EDGE_DROP_SWEEP,
        help="Comma-separated edge-drop probabilities.",
    )
    parser.add_argument(
        "--sparse-flip-sweep",
        type=_parse_sparse_flip_sweep,
        default=SPARSE_FLIP_SWEEP,
        help="Semicolon-separated p_delete:p_add:max_additions configs.",
    )
    parser.add_argument("--certified-samples", type=int, default=CERTIFIED_ACCURACY_SAMPLES)
    parser.add_argument("--local-certificate-samples", type=int, default=LOCAL_CERTIFICATE_SAMPLES)
    parser.add_argument(
        "--certificate-node-count",
        type=_parse_optional_int,
        default=CERTIFIED_ACCURACY_NODE_COUNT,
        help="Number of test nodes. Use none/all/full for the full test mask.",
    )
    parser.add_argument("--certificate-max-radius", type=int, default=CERTIFICATE_MAX_RADIUS)
    parser.add_argument("--asymmetric-max-delete", type=int, default=ASYMMETRIC_CERTIFICATE_MAX_DELETE)
    parser.add_argument("--asymmetric-max-add", type=int, default=ASYMMETRIC_CERTIFICATE_MAX_ADD)
    parser.add_argument("--nettack-targets", type=int, default=NETTACK_TARGET_COUNT, help="Nettack target nodes; 0 skips Nettack.")
    parser.add_argument(
        "--smoothing-modes",
        type=_parse_smoothing_modes,
        default=DEFAULT_SMOOTHING_MODES,
        help="Comma-separated modes: edge-drop,sparse-edge-flip,symmetric-cert,sparse-asymmetric-cert.",
    )
    return parser


def _apply_cli_config(args):
    global GLOBAL_ATTACK_BUDGETS
    global EDGE_DROP_SWEEP
    global SPARSE_FLIP_SWEEP
    global CERTIFIED_ACCURACY_SAMPLES
    global CERTIFIED_ACCURACY_NODE_COUNT
    global LOCAL_CERTIFICATE_SAMPLES
    global CERTIFICATE_MAX_RADIUS
    global ASYMMETRIC_CERTIFICATE_MAX_DELETE
    global ASYMMETRIC_CERTIFICATE_MAX_ADD
    global NETTACK_TARGET_COUNT

    GLOBAL_ATTACK_BUDGETS = args.attack_budgets
    EDGE_DROP_SWEEP = args.edge_drop_sweep
    SPARSE_FLIP_SWEEP = args.sparse_flip_sweep
    CERTIFIED_ACCURACY_SAMPLES = args.certified_samples
    CERTIFIED_ACCURACY_NODE_COUNT = args.certificate_node_count
    LOCAL_CERTIFICATE_SAMPLES = args.local_certificate_samples
    CERTIFICATE_MAX_RADIUS = args.certificate_max_radius
    ASYMMETRIC_CERTIFICATE_MAX_DELETE = args.asymmetric_max_delete
    ASYMMETRIC_CERTIFICATE_MAX_ADD = args.asymmetric_max_add
    NETTACK_TARGET_COUNT = args.nettack_targets


def _resolve_seeds(args):
    if args.seeds is not None:
        return args.seeds
    if args.seed is not None:
        return [args.seed]
    return DEFAULT_SEEDS


def main():
    args = build_arg_parser().parse_args()
    _apply_cli_config(args)
    seeds = _resolve_seeds(args)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    payloads = []
    for seed in seeds:
        seed_output_dir = output_root if len(seeds) == 1 else output_root / f"seed_{seed}"
        print(f"\n=== Running seed {seed} ===")
        payload = run_single_experiment(
            seed=seed,
            results_dir=seed_output_dir,
            dataset_name=args.dataset,
            smoothing_modes=args.smoothing_modes,
            epochs=args.epochs,
        )
        payloads.append(payload)

    if len(payloads) > 1:
        aggregate_rows = _summarize_multi_seed(payloads)
        aggregate_payload = {
            "dataset": args.dataset,
            "seeds": seeds,
            "smoothing_modes": args.smoothing_modes,
            "summary": aggregate_rows,
        }
        save_csv_rows(output_root / "multi_seed_summary.csv", aggregate_rows)
        save_json_report(output_root / "multi_seed_summary.json", aggregate_payload)
        print(f"\nSaved multi-seed summary to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
