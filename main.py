import argparse
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.attack import run_prbcd_attack
from src.models import GCN
from src.nettack import (
    choose_correct_test_nodes,
    choose_jointly_correct_test_nodes,
    predict_node,
    run_nettack_on_node,
    train_deeprobust_surrogate,
)
from src.purification import purify_edge_index_by_jaccard, purify_target_node_edges_by_jaccard
from src.reporting import (
    ensure_results_dir,
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
    train_one_epoch_with_noise,
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
SPARSE_CERTIFICATE_SWEEP = SPARSE_FLIP_SWEEP
PURIFICATION_SWEEP = [0.0, 0.01, 0.02, 0.05]
PURIFIED_CERTIFICATE_CANDIDATE_SWEEP = [
    {
        "label": "sparse-asym-0.01",
        "mode": "sparse-edge-flip",
        "p_delete": 0.01,
        "p_add": 0.000005,
        "max_additions": 64,
    },
    {
        "label": "sparse-asym-0.02",
        "mode": "sparse-edge-flip",
        "p_delete": 0.02,
        "p_add": 0.000010,
        "max_additions": 96,
    },
    {
        "label": "sparse-asym-0.05",
        "mode": "sparse-edge-flip",
        "p_delete": 0.05,
        "p_add": 0.000020,
        "max_additions": 160,
    },
    {
        "label": "balanced-sparse-0.2",
        "mode": "sparse-edge-flip",
        "p_delete": 0.2,
        "p_add": 0.2,
        "max_additions": 160,
    },
    {"label": "symmetric-0.2", "mode": "symmetric-edge-flip", "p_delete": 0.2, "p_add": 0.2, "max_additions": 160},
    {"label": "symmetric-0.3", "mode": "symmetric-edge-flip", "p_delete": 0.3, "p_add": 0.3, "max_additions": 160},
]
PURIFIED_CERTIFICATE_TARGET_COUNT = 12
PURIFICATION_ATTACK_BUDGET = 50
CERTIFIED_ACCURACY_SAMPLES = 100
CERTIFIED_ACCURACY_NODE_COUNT = 24
CONFIDENCE_Z = 1.96
LOCAL_CERTIFICATE_SAMPLES = 400
CERTIFICATE_MAX_RADIUS = 5
ASYMMETRIC_CERTIFICATE_MAX_DELETE = 5
ASYMMETRIC_CERTIFICATE_MAX_ADD = 5
SMOOTHING_SELECTION_SAMPLES = 100
SMOOTHING_CERTIFICATE_ALPHA = 0.001
FOCUS_SELECTION_P_FLIP = 0.02
FOCUS_SELECTION_SAMPLES = 100
NETTACK_TARGET_COUNT = 5
DEFAULT_SEEDS = [42]
DEFAULT_SMOOTHING_MODES = ["edge-drop", "sparse-edge-flip", "symmetric-cert", "sparse-asymmetric-cert"]
ROBUST_TRAINING_CONFIG = SPARSE_FLIP_SWEEP[1]
MATCHED_SPARSE_TRAINING_LABEL = "matched-sparse-noisy-training"
CERTIFICATE_ORIENTED_TRAINING_LABEL = "clean-warmstart-symmetric-finetune"
MATCHED_SPARSE_NOISE_CONFIG = {
    "mode": "sparse-edge-flip",
    "p_delete": ROBUST_TRAINING_CONFIG["p_delete"],
    "p_add": ROBUST_TRAINING_CONFIG["p_add"],
    "max_additions": ROBUST_TRAINING_CONFIG["max_additions"],
}
CERTIFICATE_ORIENTED_FINE_TUNE_CONFIG = {
    "mode": "symmetric-edge-flip",
    "p_delete": 0.3,
    "p_add": 0.3,
    "max_additions": 160,
}
CERTIFICATE_ORIENTED_FINE_TUNE_EPOCHS = 40
CERTIFICATE_ORIENTED_FINE_TUNE_LR = 0.005
MULTI_SEED_PURIFIED_SUMMARY_SEEDS = [7, 21, 42, 84, 99]


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
    if not values:
        return
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

    _add_mean_std(rows, "clean_test_accuracy", [payload["clean_metrics"]["test"] for payload in payloads])
    _add_mean_std(rows, "best_test_accuracy", [payload["best_test_accuracy"] for payload in payloads])

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
    _add_mean_std(rows, "edge_drop_smoothed_test_accuracy", edge_drop_values)
    _add_mean_std(rows, "sparse_edge_flip_smoothed_test_accuracy", sparse_values)

    attack_by_budget = {}
    for payload in payloads:
        for row in payload.get("global_attack_results", []):
            attack_by_budget.setdefault((("budget", int(row["budget"])),), []).append(float(row["test_accuracy"]))
    _add_grouped_mean_std(rows, "prbcd_attacked_test_accuracy", attack_by_budget)

    symmetric_cert_by_flip = {}
    for payload in payloads:
        for row in payload.get("local_certificate_accuracy_subset_sweep", []):
            symmetric_cert_by_flip.setdefault((("p_flip", float(row["p_flip"])),), []).append(
                float(row["positive_certified_accuracy"])
            )
    _add_grouped_mean_std(rows, "symmetric_positive_certified_accuracy", symmetric_cert_by_flip)

    sparse_cert_by_config = {}
    for payload in payloads:
        for row in payload.get("sparse_local_certificate_accuracy_subset_sweep", []):
            key = (
                ("p_delete", float(row["p_delete"])),
                ("p_add", float(row["p_add"])),
                ("max_additions", int(row["max_additions"])),
            )
            sparse_cert_by_config.setdefault(key, []).append(float(row["positive_certified_accuracy"]))
    _add_grouped_mean_std(rows, "sparse_positive_certified_accuracy", sparse_cert_by_config)

    robust_symmetric_by_flip = {}
    for payload in payloads:
        for row in payload.get("robust_local_certificate_accuracy_subset_sweep", []):
            robust_symmetric_by_flip.setdefault((("p_flip", float(row["p_flip"])),), []).append(
                float(row["positive_certified_accuracy"])
            )
    _add_grouped_mean_std(rows, "robust_symmetric_positive_certified_accuracy", robust_symmetric_by_flip)

    robust_sparse_by_config = {}
    for payload in payloads:
        for row in payload.get("robust_sparse_local_certificate_accuracy_subset_sweep", []):
            key = (
                ("p_delete", float(row["p_delete"])),
                ("p_add", float(row["p_add"])),
                ("max_additions", int(row["max_additions"])),
            )
            robust_sparse_by_config.setdefault(key, []).append(float(row["positive_certified_accuracy"]))
    _add_grouped_mean_std(rows, "robust_sparse_positive_certified_accuracy", robust_sparse_by_config)

    return rows


def _format_counter(counter):
    if not counter:
        return ""
    return "; ".join(f"{label}:{count}" for label, count in sorted(counter.items()))



def _print_metrics(prefix, metrics):
    print(
        f"{prefix} | "
        f"Train {metrics['train']:.4f} | "
        f"Val {metrics['val']:.4f} | "
        f"Test {metrics['test']:.4f}"
    )


def _build_additional_training_variant_specs():
    return [
        {
            "label": MATCHED_SPARSE_TRAINING_LABEL,
            "display_name": "Matched sparse-noisy robust training variant",
            "epochs": 200,
            "learning_rate": 0.01,
            "weight_decay": 5e-4,
            "warm_start_from": None,
            "train_config": {
                "variant": MATCHED_SPARSE_TRAINING_LABEL,
                **MATCHED_SPARSE_NOISE_CONFIG,
            },
        },
        {
            "label": CERTIFICATE_ORIENTED_TRAINING_LABEL,
            "display_name": "Clean warm-start symmetric fine-tuning variant",
            "epochs": CERTIFICATE_ORIENTED_FINE_TUNE_EPOCHS,
            "learning_rate": CERTIFICATE_ORIENTED_FINE_TUNE_LR,
            "weight_decay": 5e-4,
            "warm_start_from": "clean-training",
            "train_config": {
                "variant": CERTIFICATE_ORIENTED_TRAINING_LABEL,
                "track_initial_state": True,
                **CERTIFICATE_ORIENTED_FINE_TUNE_CONFIG,
            },
        },
    ]


def _summarize_certificate_rows(rows, config_keys, group_keys):
    grouped = {}

    for row in rows:
        key = tuple(row[key_name] for key_name in [*config_keys, *group_keys])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in grouped.items():
        summary_row = {}
        for key_name, key_value in zip([*config_keys, *group_keys], key):
            summary_row[key_name] = key_value

        total = len(group_rows)
        correct_rows = [row for row in group_rows if row["is_correct"]]
        non_abstained_rows = [row for row in group_rows if not row["abstained"]]
        positive_rows = [row for row in correct_rows if row["reported_certified_radius"] > 0]

        summary_row.update(
            {
                "evaluated_nodes": total,
                "correct_fraction": len(correct_rows) / total if total > 0 else 0.0,
                "abstained_fraction": sum(bool(row["abstained"]) for row in group_rows) / total if total > 0 else 0.0,
                "non_abstain_accuracy": (
                    sum(bool(row["is_correct"]) for row in non_abstained_rows) / len(non_abstained_rows)
                    if non_abstained_rows
                    else 0.0
                ),
                "positive_certified_accuracy": len(positive_rows) / total if total > 0 else 0.0,
                "certified_fraction_on_correct": len(positive_rows) / len(correct_rows) if correct_rows else 0.0,
                "mean_reported_radius_on_correct": (
                    sum(row["reported_certified_radius"] for row in correct_rows) / len(correct_rows)
                    if correct_rows
                    else 0.0
                ),
                "mean_pA_lower": sum(row["pA_lower"] for row in group_rows) / total if total > 0 else 0.0,
                "mean_smoothed_confidence": (
                    sum(row["smoothed_confidence"] for row in group_rows) / total if total > 0 else 0.0
                ),
            }
        )
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: tuple(row[key_name] for key_name in [*config_keys, *group_keys]))
    return summary_rows



def _train_model(model, data, optimizer, epochs: int = 200):
    history = []
    best_val = 0.0
    best_test = 0.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

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
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

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
    if best_epoch:
        model.load_state_dict(best_state)
        print(f"Restored best-val checkpoint from epoch {best_epoch:03d}")
    return history, best_val, best_test


def _build_model(dataset, device):
    return GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=32,
        out_channels=dataset.num_classes,
        dropout=0.5,
    ).to(device)


def _train_model_with_config(model, data, optimizer, epochs: int = 200, train_config=None):
    history = []
    train_config = train_config or {"variant": "clean"}
    variant = train_config.get("variant", "clean")
    best_val = 0.0
    best_test = 0.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

    if train_config.get("track_initial_state"):
        initial_metrics, _ = evaluate(model, data)
        best_val = float(initial_metrics["val"])
        best_test = float(initial_metrics["test"])

    def _resolve_epoch_noise_config(epoch_index):
        if "noise_schedule" in train_config:
            noise_schedule = train_config.get("noise_schedule") or []
            if not noise_schedule:
                raise ValueError(f"{variant} requires a non-empty noise_schedule")
            return noise_schedule[(epoch_index - 1) % len(noise_schedule)]
        if train_config.get("mode") and variant != "clean":
            return train_config
        return None

    for epoch in range(1, epochs + 1):
        epoch_noise_config = _resolve_epoch_noise_config(epoch)
        if epoch_noise_config is not None:
            loss = train_one_epoch_with_noise(
                model=model,
                data=data,
                optimizer=optimizer,
                mode=epoch_noise_config.get("mode", "sparse-edge-flip"),
                p_delete=epoch_noise_config.get("p_delete", 0.02),
                p_add=epoch_noise_config.get("p_add", 0.0),
                max_additions=epoch_noise_config.get("max_additions", 20000),
            )
        else:
            loss = train_one_epoch(model, data, optimizer)

        metrics, _ = evaluate(model, data)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss),
                "train": float(metrics["train"]),
                "val": float(metrics["val"]),
                "test": float(metrics["test"]),
                "variant": variant,
                "noise_mode": epoch_noise_config.get("mode", "") if epoch_noise_config is not None else "",
                "noise_p_delete": (
                    float(epoch_noise_config.get("p_delete", 0.0)) if epoch_noise_config is not None else 0.0
                ),
                "noise_p_add": float(epoch_noise_config.get("p_add", 0.0)) if epoch_noise_config is not None else 0.0,
                "noise_max_additions": (
                    int(epoch_noise_config.get("max_additions", 0)) if epoch_noise_config is not None else 0
                ),
            }
        )

        if metrics["val"] > best_val:
            best_val = metrics["val"]
            best_test = metrics["test"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if best_epoch:
        print(f"Restored {variant} best-val checkpoint from epoch {best_epoch:03d}")
    elif train_config.get("track_initial_state"):
        print(f"Restored {variant} initial checkpoint after no fine-tuning improvement")
    return history, best_val, best_test


def _run_variant_global_attack_evaluation(model, data, device, variant_label, heading):
    attack_rows = []
    print(f"\n=== {heading} ===")
    for budget in GLOBAL_ATTACK_BUDGETS:
        attacked_edge_index, flipped_edges = run_prbcd_attack(model, data, budget, device)
        attacked_metrics, _ = evaluate_with_edge_index(model, data, attacked_edge_index)
        num_flips = flipped_edges.size(1) if flipped_edges.numel() > 0 else 0
        attack_rows.append(
            {
                "variant": variant_label,
                "budget": budget,
                "test_accuracy": float(attacked_metrics["test"]),
                "flipped_edges": int(num_flips),
            }
        )
        print(
            f"Budget {budget:>3d} | "
            f"{variant_label} attacked test accuracy: {attacked_metrics['test']:.4f} | "
            f"Flipped edges: {num_flips}"
        )

    return attack_rows


def _evaluate_symmetric_certificate_subset(
    model,
    data,
    certificate_subset_nodes,
    certificate_subset_metadata,
    curve_label_prefix="",
):
    summary_rows = []
    curve_rows = []
    curves = []
    subset_rows = []

    for p_flip in CERTIFIED_ACCURACY_SWEEP:
        per_node_certificates = []
        for node_idx in certificate_subset_nodes:
            certificate, _, _ = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=data.edge_index,
                node_idx=node_idx,
                num_samples=CERTIFIED_ACCURACY_SAMPLES,
                selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
                certification_num_samples=CERTIFIED_ACCURACY_SAMPLES,
                mode="symmetric-edge-flip",
                p_delete=p_flip,
                certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
                certificate_beta=p_flip,
                certificate_max_radius=CERTIFICATE_MAX_RADIUS,
            )
            certificate["node_idx"] = int(node_idx)
            certificate["p_flip"] = float(p_flip)
            per_node_certificates.append(certificate)
            subset_rows.append(
                {
                    "p_flip": float(p_flip),
                    "node_idx": int(node_idx),
                    "degree": int(certificate_subset_metadata[node_idx]["degree"]),
                    "degree_bucket": int(certificate_subset_metadata[node_idx]["degree_bucket"]),
                    "clean_confidence": float(certificate_subset_metadata[node_idx]["clean_confidence"]),
                    "clean_margin": float(certificate_subset_metadata[node_idx]["clean_margin"]),
                    "margin_bucket": int(certificate_subset_metadata[node_idx]["margin_bucket"]),
                    "predicted_class": int(certificate["predicted_class"]),
                    "true_label": int(certificate["true_label"]),
                    "abstained": bool(certificate["abstained"]),
                    "is_correct": bool(certificate["is_correct"]),
                    "certified_radius": int(certificate["certified_radius"] or 0),
                    "reported_certified_radius": int(certificate["reported_certified_radius"]),
                    "reported_asymmetric_total_radius": int(certificate["reported_asymmetric_total_radius"]),
                    "reported_asymmetric_delete_budget": int(certificate["reported_asymmetric_delete_budget"]),
                    "reported_asymmetric_add_budget": int(certificate["reported_asymmetric_add_budget"]),
                    "reported_runner_up_asymmetric_total_radius": int(
                        certificate["reported_runner_up_asymmetric_total_radius"]
                    ),
                    "smoothed_confidence": float(certificate["pA_hat"]),
                    "pA_lower": float(certificate["pA_lower"]),
                }
            )

        correct_nodes = sum(certificate["is_correct"] for certificate in per_node_certificates)
        reported_radii = [
            certificate["reported_certified_radius"]
            for certificate in per_node_certificates
            if certificate["is_correct"]
        ]
        certified_accuracy_curve = build_certified_accuracy_curve(
            per_node_certificates,
            max_radius=CERTIFICATE_MAX_RADIUS,
        )

        row = {
            "p_flip": float(p_flip),
            "evaluated_nodes": int(len(per_node_certificates)),
            "correct_fraction": correct_nodes / len(per_node_certificates) if per_node_certificates else 0.0,
            "positive_certified_accuracy": sum(radius > 0 for radius in reported_radii) / len(per_node_certificates)
            if per_node_certificates
            else 0.0,
            "certified_fraction_on_correct": sum(radius > 0 for radius in reported_radii) / correct_nodes
            if correct_nodes > 0
            else 0.0,
            "mean_certified_radius_on_correct": sum(reported_radii) / len(reported_radii) if reported_radii else 0.0,
        }
        summary_rows.append(row)

        per_config_curve_rows = []
        for point in certified_accuracy_curve:
            curve_row = {
                "p_flip": float(p_flip),
                "radius": int(point["radius"]),
                "certified_accuracy": float(point["certified_accuracy"]),
            }
            curve_rows.append(curve_row)
            per_config_curve_rows.append(
                {
                    "radius": curve_row["radius"],
                    "certified_accuracy": curve_row["certified_accuracy"],
                }
            )

        curves.append(
            {
                "label": f"{curve_label_prefix}symmetric p_flip={p_flip:.2f}".strip(),
                "rows": per_config_curve_rows,
            }
        )

        print(
            f"p_flip={p_flip:.2f} | "
            f"Subset accuracy {row['correct_fraction']:.4f} | "
            f"Certified acc @ r>=1 {row['positive_certified_accuracy']:.4f} | "
            f"Mean radius on correct {row['mean_certified_radius_on_correct']:.3f}"
        )

    return {
        "summary_rows": summary_rows,
        "curve_rows": curve_rows,
        "curves": curves,
        "subset_rows": subset_rows,
    }


def _evaluate_sparse_certificate_subset(
    model,
    data,
    certificate_subset_nodes,
    certificate_subset_metadata,
    curve_label_prefix="",
):
    summary_rows = []
    curve_rows = []
    curves = []
    subset_rows = []

    for config in SPARSE_CERTIFICATE_SWEEP:
        per_node_certificates = []
        p_delete = config["p_delete"]
        p_add = config["p_add"]
        max_additions = config["max_additions"]

        for node_idx in certificate_subset_nodes:
            certificate, _, _ = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=data.edge_index,
                node_idx=node_idx,
                num_samples=CERTIFIED_ACCURACY_SAMPLES,
                selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
                certification_num_samples=CERTIFIED_ACCURACY_SAMPLES,
                mode="sparse-edge-flip",
                p_delete=p_delete,
                p_add=p_add,
                max_additions=max_additions,
                certificate_p_delete=p_delete,
                certificate_p_add=p_add,
                certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
                certificate_max_radius=CERTIFICATE_MAX_RADIUS,
                certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
            )
            certificate["node_idx"] = int(node_idx)
            per_node_certificates.append(certificate)
            subset_rows.append(
                {
                    "p_delete": float(p_delete),
                    "p_add": float(p_add),
                    "max_additions": int(max_additions),
                    "node_idx": int(node_idx),
                    "degree": int(certificate_subset_metadata[node_idx]["degree"]),
                    "degree_bucket": int(certificate_subset_metadata[node_idx]["degree_bucket"]),
                    "clean_confidence": float(certificate_subset_metadata[node_idx]["clean_confidence"]),
                    "clean_margin": float(certificate_subset_metadata[node_idx]["clean_margin"]),
                    "margin_bucket": int(certificate_subset_metadata[node_idx]["margin_bucket"]),
                    "predicted_class": int(certificate["predicted_class"]),
                    "true_label": int(certificate["true_label"]),
                    "abstained": bool(certificate["abstained"]),
                    "is_correct": bool(certificate["is_correct"]),
                    "certified_radius": int(certificate["certified_radius"] or 0),
                    "reported_certified_radius": int(certificate["reported_certified_radius"]),
                    "reported_asymmetric_total_radius": int(certificate["reported_asymmetric_total_radius"]),
                    "reported_asymmetric_delete_budget": int(certificate["reported_asymmetric_delete_budget"]),
                    "reported_asymmetric_add_budget": int(certificate["reported_asymmetric_add_budget"]),
                    "reported_runner_up_asymmetric_total_radius": int(
                        certificate["reported_runner_up_asymmetric_total_radius"]
                    ),
                    "smoothed_confidence": float(certificate["pA_hat"]),
                    "pA_lower": float(certificate["pA_lower"]),
                }
            )

        correct_nodes = sum(certificate["is_correct"] for certificate in per_node_certificates)
        reported_radii = [
            certificate["reported_certified_radius"]
            for certificate in per_node_certificates
            if certificate["is_correct"]
        ]
        certified_accuracy_curve = build_certified_accuracy_curve(
            per_node_certificates,
            max_radius=CERTIFICATE_MAX_RADIUS,
        )

        row = {
            "p_delete": float(p_delete),
            "p_add": float(p_add),
            "max_additions": int(max_additions),
            "evaluated_nodes": int(len(per_node_certificates)),
            "correct_fraction": correct_nodes / len(per_node_certificates) if per_node_certificates else 0.0,
            "positive_certified_accuracy": sum(radius > 0 for radius in reported_radii) / len(per_node_certificates)
            if per_node_certificates
            else 0.0,
            "certified_fraction_on_correct": sum(radius > 0 for radius in reported_radii) / correct_nodes
            if correct_nodes > 0
            else 0.0,
            "mean_certified_radius_on_correct": sum(reported_radii) / len(reported_radii) if reported_radii else 0.0,
        }
        summary_rows.append(row)

        per_config_curve_rows = []
        for point in certified_accuracy_curve:
            curve_row = {
                "p_delete": float(p_delete),
                "p_add": float(p_add),
                "max_additions": int(max_additions),
                "radius": int(point["radius"]),
                "certified_accuracy": float(point["certified_accuracy"]),
            }
            curve_rows.append(curve_row)
            per_config_curve_rows.append(
                {
                    "radius": curve_row["radius"],
                    "certified_accuracy": curve_row["certified_accuracy"],
                }
            )

        curves.append(
            {
                "label": f"{curve_label_prefix}sparse p_del={p_delete:.3f}, p_add={p_add:.6f}".strip(),
                "rows": per_config_curve_rows,
            }
        )

        print(
            f"p_delete={p_delete:.3f}, p_add={p_add:.6f} | "
            f"Subset accuracy {row['correct_fraction']:.4f} | "
            f"Certified acc @ r>=1 {row['positive_certified_accuracy']:.4f} | "
            f"Mean radius on correct {row['mean_certified_radius_on_correct']:.3f}"
        )

    return {
        "summary_rows": summary_rows,
        "curve_rows": curve_rows,
        "curves": curves,
        "subset_rows": subset_rows,
    }


def _tag_certificate_summary_rows(rows, *, variant, certificate_family, subset_name):
    tagged_rows = []
    for row in rows:
        tagged_rows.append(
            {
                "subset_name": subset_name,
                "variant": variant,
                "certificate_family": certificate_family,
                **row,
            }
        )
    return tagged_rows



def _select_focus_result(model, data, nettack_results):
    candidates = [result for result in nettack_results if result["success"]]
    if not candidates:
        candidates = nettack_results
    if not candidates:
        return None, None

    beta = FOCUS_SELECTION_P_FLIP
    best_bundle = None
    best_score = None

    for result in candidates:
        clean_certificate, _, _ = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=data.edge_index,
            node_idx=result["target_node"],
            num_samples=FOCUS_SELECTION_SAMPLES,
            selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
            certification_num_samples=FOCUS_SELECTION_SAMPLES,
            mode="symmetric-edge-flip",
            p_delete=FOCUS_SELECTION_P_FLIP,
            certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
            certificate_beta=beta,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
        )
        attack_certificate, _, _ = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=result["attacked_edge_index"],
            node_idx=result["target_node"],
            num_samples=FOCUS_SELECTION_SAMPLES,
            selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
            certification_num_samples=FOCUS_SELECTION_SAMPLES,
            mode="symmetric-edge-flip",
            p_delete=FOCUS_SELECTION_P_FLIP,
            certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
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


def _collect_nettack_result_pool(
    target_model,
    data,
    surrogate,
    adj,
    features,
    labels,
    target_count,
    device="cpu",
    initial_results=None,
    initial_count=5,
):
    attempted_results = list(initial_results or [])
    successful_results = [result for result in attempted_results if result["success"]]
    attempted_nodes = {int(result["target_node"]) for result in attempted_results}

    if not attempted_results:
        initial_nodes = choose_correct_test_nodes(
            target_model,
            data,
            k=min(initial_count, target_count),
            strategy="stratified",
        )
        for target_node in initial_nodes:
            result = run_nettack_on_node(
                target_model=target_model,
                data=data,
                surrogate=surrogate,
                adj=adj,
                features=features,
                labels=labels,
                target_node=target_node,
                n_perturbations=3,
                device=device,
            )
            attempted_results.append(result)
            attempted_nodes.add(int(target_node))
            if result["success"]:
                successful_results.append(result)

    if len(successful_results) < target_count:
        expanded_nodes = choose_correct_test_nodes(
            target_model,
            data,
            k=target_count,
            strategy="stratified",
        )
        for target_node in expanded_nodes:
            if target_node in attempted_nodes:
                continue
            result = run_nettack_on_node(
                target_model=target_model,
                data=data,
                surrogate=surrogate,
                adj=adj,
                features=features,
                labels=labels,
                target_node=target_node,
                n_perturbations=3,
                device=device,
            )
            attempted_results.append(result)
            attempted_nodes.add(int(target_node))
            if result["success"]:
                successful_results.append(result)
            if len(successful_results) >= target_count:
                break

    selected_results = successful_results if successful_results else list(attempted_results)
    return {
        "attempted_results": attempted_results,
        "selected_results": selected_results,
        "attempted_nodes": int(len(attempted_nodes)),
        "successful_attacks": int(len(successful_results)),
        "used_fallback_pool": int(not successful_results),
    }


def _evaluate_purified_certificate_sweep(
    model,
    data,
    purified_certificate_results,
    variant_name,
    attack_variant,
    emit_logs=True,
    log_prefix="",
):
    candidate_rows = []
    candidate_summary = []
    oracle_rows = []
    oracle_summary = []

    for threshold in PURIFICATION_SWEEP[1:]:
        per_threshold_candidate_rows = []
        per_threshold_best_rows = []
        for result in purified_certificate_results:
            purified_edge_index, purification_stats = purify_target_node_edges_by_jaccard(
                x=data.x,
                edge_index=result["attacked_edge_index"],
                target_node=result["target_node"],
                threshold=threshold,
            )
            purified_pred, _ = predict_node(
                model=model,
                data=data,
                edge_index=purified_edge_index,
                node_idx=result["target_node"],
            )

            per_target_candidate_rows = []
            for config in PURIFIED_CERTIFICATE_CANDIDATE_SWEEP:
                certificate_kwargs = {
                    "certificate_alpha": SMOOTHING_CERTIFICATE_ALPHA,
                    "certificate_max_radius": CERTIFICATE_MAX_RADIUS,
                }
                if config["mode"] == "symmetric-edge-flip":
                    certificate_kwargs["certificate_beta"] = config["p_delete"]
                else:
                    certificate_kwargs["certificate_p_delete"] = config["p_delete"]
                    certificate_kwargs["certificate_p_add"] = config["p_add"]
                    certificate_kwargs["certificate_max_delete"] = ASYMMETRIC_CERTIFICATE_MAX_DELETE
                    certificate_kwargs["certificate_max_add"] = ASYMMETRIC_CERTIFICATE_MAX_ADD

                certificate, _, smoothed_pred = evaluate_smoothed_node_with_edge_index(
                    model=model,
                    data=data,
                    edge_index=purified_edge_index,
                    node_idx=result["target_node"],
                    num_samples=LOCAL_CERTIFICATE_SAMPLES,
                    selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
                    certification_num_samples=LOCAL_CERTIFICATE_SAMPLES,
                    mode=config["mode"],
                    p_delete=config["p_delete"],
                    p_add=config["p_add"],
                    max_additions=config["max_additions"],
                    **certificate_kwargs,
                )
                row = {
                    "summary_type": "fixed-config",
                    "model_variant": variant_name,
                    "attack_variant": attack_variant,
                    "threshold": float(threshold),
                    "config_label": config["label"],
                    "mode": config["mode"],
                    "p_delete": float(config["p_delete"]),
                    "p_add": float(config["p_add"]),
                    "max_additions": int(config["max_additions"]),
                    "target_node": int(result["target_node"]),
                    "true_label": int(result["true_label"]),
                    "purified_pred": int(purified_pred),
                    "purified_is_correct": int(purified_pred == result["true_label"]),
                    "smoothed_pred": int(smoothed_pred),
                    "is_correct": int(certificate["is_correct"]),
                    "reported_certified_radius": int(certificate["reported_certified_radius"]),
                    "reported_asymmetric_total_radius": int(certificate["reported_asymmetric_total_radius"]),
                    "reported_asymmetric_delete_budget": int(certificate["reported_asymmetric_delete_budget"]),
                    "reported_asymmetric_add_budget": int(certificate["reported_asymmetric_add_budget"]),
                    "reported_runner_up_asymmetric_total_radius": int(
                        certificate["reported_runner_up_asymmetric_total_radius"]
                    ),
                    "pA_lower": float(certificate["pA_lower"]),
                    "target_edge_retention": float(purification_stats["target_edge_retention"]),
                }
                per_target_candidate_rows.append(row)
                per_threshold_candidate_rows.append(row)
                candidate_rows.append(row)

            selected_row = dict(per_target_candidate_rows[0])
            selection_status = "fallback-low-noise"
            best_correct_row = None
            for row in per_target_candidate_rows:
                if not row["is_correct"]:
                    continue
                if best_correct_row is None:
                    best_correct_row = row
                    continue
                if row["reported_certified_radius"] > best_correct_row["reported_certified_radius"]:
                    best_correct_row = row
                    continue
                if (
                    row["reported_certified_radius"] == best_correct_row["reported_certified_radius"]
                    and row["pA_lower"] > best_correct_row["pA_lower"]
                ):
                    best_correct_row = row

            if best_correct_row is not None:
                selected_row = dict(best_correct_row)
                selection_status = "best-correct-radius"

            selected_row["summary_type"] = "oracle-best-correct"
            selected_row["selection_status"] = selection_status
            selected_row["had_correct_candidate"] = int(best_correct_row is not None)
            oracle_rows.append(selected_row)
            per_threshold_best_rows.append(selected_row)

        for config in PURIFIED_CERTIFICATE_CANDIDATE_SWEEP:
            per_config_rows = [row for row in per_threshold_candidate_rows if row["config_label"] == config["label"]]
            candidate_summary.append(
                {
                    "summary_type": "fixed-config",
                    "model_variant": variant_name,
                    "attack_variant": attack_variant,
                    "threshold": float(threshold),
                    "config_label": config["label"],
                    "mode": config["mode"],
                    "p_delete": float(config["p_delete"]),
                    "p_add": float(config["p_add"]),
                    "max_additions": int(config["max_additions"]),
                    "evaluated_nodes": int(len(per_config_rows)),
                    "purified_plain_correct_fraction": (
                        sum(row["purified_is_correct"] for row in per_config_rows) / len(per_config_rows)
                        if per_config_rows
                        else 0.0
                    ),
                    "correct_fraction": (
                        sum(row["is_correct"] for row in per_config_rows) / len(per_config_rows)
                        if per_config_rows
                        else 0.0
                    ),
                    "positive_certified_fraction": (
                        sum(row["reported_certified_radius"] > 0 for row in per_config_rows) / len(per_config_rows)
                        if per_config_rows
                        else 0.0
                    ),
                    "max_reported_radius": max(
                        (row["reported_certified_radius"] for row in per_config_rows),
                        default=0,
                    ),
                    "mean_pA_lower": (
                        sum(row["pA_lower"] for row in per_config_rows) / len(per_config_rows)
                        if per_config_rows
                        else 0.0
                    ),
                }
            )

        selected_config_counts = Counter(
            row["config_label"] for row in per_threshold_best_rows if row["had_correct_candidate"]
        )
        summary_row = {
            "summary_type": "oracle-best-correct",
            "model_variant": variant_name,
            "attack_variant": attack_variant,
            "threshold": float(threshold),
            "evaluated_nodes": int(len(per_threshold_best_rows)),
            "purified_plain_correct_fraction": (
                sum(row["purified_is_correct"] for row in per_threshold_best_rows) / len(per_threshold_best_rows)
                if per_threshold_best_rows
                else 0.0
            ),
            "correct_fraction": (
                sum(row["is_correct"] for row in per_threshold_best_rows) / len(per_threshold_best_rows)
                if per_threshold_best_rows
                else 0.0
            ),
            "positive_certified_fraction": (
                sum(row["reported_certified_radius"] > 0 for row in per_threshold_best_rows)
                / len(per_threshold_best_rows)
                if per_threshold_best_rows
                else 0.0
            ),
            "mean_reported_radius": (
                sum(row["reported_certified_radius"] for row in per_threshold_best_rows)
                / len(per_threshold_best_rows)
                if per_threshold_best_rows
                else 0.0
            ),
            "max_reported_radius": max(
                (row["reported_certified_radius"] for row in per_threshold_best_rows),
                default=0,
            ),
            "mean_pA_lower": (
                sum(row["pA_lower"] for row in per_threshold_best_rows) / len(per_threshold_best_rows)
                if per_threshold_best_rows
                else 0.0
            ),
            "mean_target_edge_retention": (
                sum(row["target_edge_retention"] for row in per_threshold_best_rows) / len(per_threshold_best_rows)
                if per_threshold_best_rows
                else 0.0
            ),
            "nodes_with_correct_config": int(sum(row["had_correct_candidate"] for row in per_threshold_best_rows)),
            "nodes_without_correct_config": int(
                sum(1 - row["had_correct_candidate"] for row in per_threshold_best_rows)
            ),
            "correct_selected_config_counts": _format_counter(selected_config_counts),
        }
        oracle_summary.append(summary_row)
        if emit_logs:
            print(
                f"{log_prefix}variant={variant_name} | thr={threshold:.3f} | Oracle best-correct | "
                f"Plain {summary_row['purified_plain_correct_fraction']:.4f} | "
                f"Smooth-correct {summary_row['correct_fraction']:.4f} | "
                f"Certified>0 {summary_row['positive_certified_fraction']:.4f} | "
                f"Max radius {summary_row['max_reported_radius']} | "
                f"Selected {summary_row['correct_selected_config_counts'] or 'none'}"
            )

    return {
        "candidate_rows": candidate_rows,
        "candidate_summary": candidate_summary,
        "oracle_rows": oracle_rows,
        "oracle_summary": oracle_summary,
    }


def _evaluate_focus_purification_case(
    model,
    data,
    focus_result,
    focus_probe,
    variant_name,
    attack_variant,
    sparse_config,
    emit_logs=True,
):
    if focus_result is None or focus_probe is None:
        return None, []

    node_idx = focus_result["target_node"]
    attacked_edge_index = focus_result["attacked_edge_index"]
    focus_summary = {
        "model_variant": variant_name,
        "attack_variant": attack_variant,
        "target_node": int(node_idx),
        "true_label": int(focus_result["true_label"]),
        "clean_probe_correct": bool(focus_probe["clean_probe"]["is_correct"]),
        "attack_probe_correct": bool(focus_probe["attack_probe"]["is_correct"]),
        "clean_probe_radius": int(focus_probe["clean_probe"]["reported_certified_radius"]),
        "attack_probe_radius": int(focus_probe["attack_probe"]["reported_certified_radius"]),
    }
    focus_rows = []

    if emit_logs:
        print(
            f"\nSelected target node {node_idx} for the {variant_name} certificate case study | "
            f"Clean probe correct: {focus_probe['clean_probe']['is_correct']} | "
            f"Attack probe correct: {focus_probe['attack_probe']['is_correct']}"
        )
        print(f"\n=== {variant_name} focus-node target purification sparse certificate sweep ===")

    for threshold in PURIFICATION_SWEEP:
        purified_focus_edge_index, purification_stats = purify_target_node_edges_by_jaccard(
            x=data.x,
            edge_index=attacked_edge_index,
            target_node=node_idx,
            threshold=threshold,
        )
        purified_plain_pred, _ = predict_node(
            model=model,
            data=data,
            edge_index=purified_focus_edge_index,
            node_idx=node_idx,
        )
        purified_certificate, _, purified_smoothed_pred = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=purified_focus_edge_index,
            node_idx=node_idx,
            num_samples=LOCAL_CERTIFICATE_SAMPLES,
            selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
            certification_num_samples=LOCAL_CERTIFICATE_SAMPLES,
            mode="sparse-edge-flip",
            p_delete=sparse_config["p_delete"],
            p_add=sparse_config["p_add"],
            max_additions=sparse_config["max_additions"],
            certificate_p_delete=sparse_config["p_delete"],
            certificate_p_add=sparse_config["p_add"],
            certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
            certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
            certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
        )

        row = {
            "model_variant": variant_name,
            "attack_variant": attack_variant,
            "threshold": float(threshold),
            "target_node": int(node_idx),
            "true_label": int(data.y[node_idx].item()),
            "plain_pred": int(purified_plain_pred),
            "plain_is_correct": int(purified_plain_pred == int(data.y[node_idx].item())),
            "smoothed_pred": int(purified_smoothed_pred),
            "smoothed_is_correct": int(purified_certificate["is_correct"]),
            "reported_certified_radius": int(purified_certificate["reported_certified_radius"]),
            "reported_asymmetric_total_radius": int(purified_certificate["reported_asymmetric_total_radius"]),
            "reported_asymmetric_delete_budget": int(purified_certificate["reported_asymmetric_delete_budget"]),
            "reported_asymmetric_add_budget": int(purified_certificate["reported_asymmetric_add_budget"]),
            "reported_runner_up_asymmetric_total_radius": int(
                purified_certificate["reported_runner_up_asymmetric_total_radius"]
            ),
            "pA_lower": float(purified_certificate["pA_lower"]),
            "target_edge_retention": float(purification_stats["target_edge_retention"]),
        }
        focus_rows.append(row)

        if emit_logs:
            print(
                f"threshold={threshold:.3f} | "
                f"Plain pred {purified_plain_pred} | "
                f"Smoothed pred {purified_smoothed_pred} | "
                f"Correct {purified_certificate['is_correct']} | "
                f"Radius {purified_certificate['reported_certified_radius']} | "
                f"pA_lower {purified_certificate['pA_lower']:.3f} | "
                f"Target keep {purification_stats['target_edge_retention']:.3f}"
            )

    return focus_summary, focus_rows


def _evaluate_focus_legacy_diagnostics(
    model,
    data,
    focus_result,
    variant_name,
    attack_variant,
    emit_logs=True,
):
    if focus_result is None:
        return {
            "edge_drop_rows": [],
            "sparse_flip_rows": [],
            "symmetric_certificate_rows": [],
            "sparse_certificate_rows": [],
        }

    node_idx = focus_result["target_node"]
    attacked_edge_index = focus_result["attacked_edge_index"]
    true_label = int(data.y[node_idx].item())
    edge_drop_rows = []
    sparse_flip_rows = []
    symmetric_certificate_rows = []
    sparse_certificate_rows = []

    if emit_logs:
        print(f"\n=== {variant_name} focus edge-drop smoothing sweep ===")

    for p_delete in EDGE_DROP_SWEEP:
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
        row = {
            "model_variant": variant_name,
            "attack_variant": attack_variant,
            "target_node": int(node_idx),
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
        if emit_logs:
            print(
                f"p_delete={p_delete:.2f} | "
                f"Clean acc {clean_metrics_s['test']:.4f} | "
                f"Attack acc {attack_metrics_s['test']:.4f} | "
                f"Clean margin {clean_cert['margin']:.3f} | "
                f"Attack margin {attack_cert['margin']:.3f}"
            )

    if emit_logs:
        print(f"\n=== {variant_name} focus sparse edge-flip smoothing sweep ===")

    for config in SPARSE_FLIP_SWEEP:
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
        row = {
            "model_variant": variant_name,
            "attack_variant": attack_variant,
            "target_node": int(node_idx),
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
        if emit_logs:
            print(
                f"p_delete={p_delete:.3f}, p_add={p_add:.6f} | "
                f"Clean acc {clean_metrics_s['test']:.4f} | "
                f"Attack acc {attack_metrics_s['test']:.4f}"
            )

    if emit_logs:
        print(f"\n=== {variant_name} focus symmetric certificate sweep ===")

    for p_flip in LOCAL_CERTIFICATE_SWEEP:
        clean_certificate, _, clean_pred = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=data.edge_index,
            node_idx=node_idx,
            num_samples=LOCAL_CERTIFICATE_SAMPLES,
            selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
            certification_num_samples=LOCAL_CERTIFICATE_SAMPLES,
            mode="symmetric-edge-flip",
            p_delete=p_flip,
            certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
            certificate_beta=p_flip,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
        )
        attack_certificate, _, attack_pred = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=attacked_edge_index,
            node_idx=node_idx,
            num_samples=LOCAL_CERTIFICATE_SAMPLES,
            selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
            certification_num_samples=LOCAL_CERTIFICATE_SAMPLES,
            mode="symmetric-edge-flip",
            p_delete=p_flip,
            certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
            certificate_beta=p_flip,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
        )

        row = {
            "model_variant": variant_name,
            "attack_variant": attack_variant,
            "target_node": int(node_idx),
            "p_flip": float(p_flip),
            "beta": float(p_flip),
            "clean_abstained": bool(clean_certificate["abstained"]),
            "attacked_abstained": bool(attack_certificate["abstained"]),
            "clean_certified_radius": int(clean_certificate["certified_radius"] or 0),
            "attacked_certified_radius": int(attack_certificate["certified_radius"] or 0),
            "clean_reported_certified_radius": int(clean_certificate["reported_certified_radius"]),
            "attacked_reported_certified_radius": int(attack_certificate["reported_certified_radius"]),
            "clean_reported_asymmetric_total_radius": int(clean_certificate["reported_asymmetric_total_radius"]),
            "attacked_reported_asymmetric_total_radius": int(attack_certificate["reported_asymmetric_total_radius"]),
            "clean_reported_asymmetric_delete_budget": int(clean_certificate["reported_asymmetric_delete_budget"]),
            "attacked_reported_asymmetric_delete_budget": int(attack_certificate["reported_asymmetric_delete_budget"]),
            "clean_reported_asymmetric_add_budget": int(clean_certificate["reported_asymmetric_add_budget"]),
            "attacked_reported_asymmetric_add_budget": int(attack_certificate["reported_asymmetric_add_budget"]),
            "clean_reported_runner_up_asymmetric_total_radius": int(
                clean_certificate["reported_runner_up_asymmetric_total_radius"]
            ),
            "attacked_reported_runner_up_asymmetric_total_radius": int(
                attack_certificate["reported_runner_up_asymmetric_total_radius"]
            ),
            "clean_runner_up_radius": int(clean_certificate["runner_up_certified_radius"] or 0),
            "attacked_runner_up_radius": int(attack_certificate["runner_up_certified_radius"] or 0),
            "clean_pA_lower": float(clean_certificate["pA_lower"]),
            "attacked_pA_lower": float(attack_certificate["pA_lower"]),
            "clean_pred": int(clean_pred),
            "attacked_pred": int(attack_pred),
            "clean_is_correct": bool(clean_certificate["is_correct"]),
            "attacked_is_correct": bool(attack_certificate["is_correct"]),
            "true_label": true_label,
        }
        symmetric_certificate_rows.append(row)
        if emit_logs:
            print(
                f"p_flip={p_flip:.2f} | "
                f"Clean reportable radius {clean_certificate['reported_certified_radius']} | "
                f"Attack reportable radius {attack_certificate['reported_certified_radius']} | "
                f"Clean pA_lower {clean_certificate['pA_lower']:.3f} | "
                f"Attack pA_lower {attack_certificate['pA_lower']:.3f}"
            )

    if emit_logs:
        print(f"\n=== {variant_name} focus sparse certificate sweep ===")

    for config in SPARSE_CERTIFICATE_SWEEP:
        p_delete = config["p_delete"]
        p_add = config["p_add"]
        max_additions = config["max_additions"]

        clean_certificate, _, clean_pred = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=data.edge_index,
            node_idx=node_idx,
            num_samples=LOCAL_CERTIFICATE_SAMPLES,
            selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
            certification_num_samples=LOCAL_CERTIFICATE_SAMPLES,
            mode="sparse-edge-flip",
            p_delete=p_delete,
            p_add=p_add,
            max_additions=max_additions,
            certificate_p_delete=p_delete,
            certificate_p_add=p_add,
            certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
            certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
            certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
        )
        attack_certificate, _, attack_pred = evaluate_smoothed_node_with_edge_index(
            model=model,
            data=data,
            edge_index=attacked_edge_index,
            node_idx=node_idx,
            num_samples=LOCAL_CERTIFICATE_SAMPLES,
            selection_num_samples=SMOOTHING_SELECTION_SAMPLES,
            certification_num_samples=LOCAL_CERTIFICATE_SAMPLES,
            mode="sparse-edge-flip",
            p_delete=p_delete,
            p_add=p_add,
            max_additions=max_additions,
            certificate_p_delete=p_delete,
            certificate_p_add=p_add,
            certificate_alpha=SMOOTHING_CERTIFICATE_ALPHA,
            certificate_max_radius=CERTIFICATE_MAX_RADIUS,
            certificate_max_delete=ASYMMETRIC_CERTIFICATE_MAX_DELETE,
            certificate_max_add=ASYMMETRIC_CERTIFICATE_MAX_ADD,
        )

        row = {
            "model_variant": variant_name,
            "attack_variant": attack_variant,
            "target_node": int(node_idx),
            "p_delete": float(p_delete),
            "p_add": float(p_add),
            "max_additions": int(max_additions),
            "clean_abstained": bool(clean_certificate["abstained"]),
            "attacked_abstained": bool(attack_certificate["abstained"]),
            "clean_certified_radius": int(clean_certificate["certified_radius"] or 0),
            "attacked_certified_radius": int(attack_certificate["certified_radius"] or 0),
            "clean_reported_certified_radius": int(clean_certificate["reported_certified_radius"]),
            "attacked_reported_certified_radius": int(attack_certificate["reported_certified_radius"]),
            "clean_reported_asymmetric_total_radius": int(clean_certificate["reported_asymmetric_total_radius"]),
            "attacked_reported_asymmetric_total_radius": int(attack_certificate["reported_asymmetric_total_radius"]),
            "clean_reported_asymmetric_delete_budget": int(clean_certificate["reported_asymmetric_delete_budget"]),
            "attacked_reported_asymmetric_delete_budget": int(attack_certificate["reported_asymmetric_delete_budget"]),
            "clean_reported_asymmetric_add_budget": int(clean_certificate["reported_asymmetric_add_budget"]),
            "attacked_reported_asymmetric_add_budget": int(attack_certificate["reported_asymmetric_add_budget"]),
            "clean_reported_runner_up_asymmetric_total_radius": int(
                clean_certificate["reported_runner_up_asymmetric_total_radius"]
            ),
            "attacked_reported_runner_up_asymmetric_total_radius": int(
                attack_certificate["reported_runner_up_asymmetric_total_radius"]
            ),
            "clean_runner_up_radius": int(clean_certificate["runner_up_certified_radius"] or 0),
            "attacked_runner_up_radius": int(attack_certificate["runner_up_certified_radius"] or 0),
            "clean_pA_lower": float(clean_certificate["pA_lower"]),
            "attacked_pA_lower": float(attack_certificate["pA_lower"]),
            "clean_pred": int(clean_pred),
            "attacked_pred": int(attack_pred),
            "clean_is_correct": bool(clean_certificate["is_correct"]),
            "attacked_is_correct": bool(attack_certificate["is_correct"]),
            "true_label": true_label,
        }
        sparse_certificate_rows.append(row)
        if emit_logs:
            print(
                f"p_delete={p_delete:.3f}, p_add={p_add:.6f} | "
                f"Clean reportable radius {clean_certificate['reported_certified_radius']} | "
                f"Attack reportable radius {attack_certificate['reported_certified_radius']} | "
                f"Clean pA_lower {clean_certificate['pA_lower']:.3f} | "
                f"Attack pA_lower {attack_certificate['pA_lower']:.3f}"
            )

    return {
        "edge_drop_rows": edge_drop_rows,
        "sparse_flip_rows": sparse_flip_rows,
        "symmetric_certificate_rows": symmetric_certificate_rows,
        "sparse_certificate_rows": sparse_certificate_rows,
    }


def _aggregate_multiseed_numeric_rows(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in grouped.items():
        summary_row = {key_name: key_value for key_name, key_value in zip(group_keys, key)}
        num_seeds = int(len(group_rows))
        summary_row["num_seeds"] = num_seeds
        summary_row["seeds"] = "; ".join(str(row["seed"]) for row in sorted(group_rows, key=lambda row: row["seed"]))
        for metric_key in metric_keys:
            values = np.array([float(row[metric_key]) for row in group_rows], dtype=float)
            mean_value = float(values.mean())
            population_std = float(values.std())
            sample_std = float(values.std(ddof=1)) if num_seeds > 1 else 0.0
            sem = sample_std / np.sqrt(num_seeds) if num_seeds > 1 else 0.0
            ci95_half_width = 1.96 * sem
            summary_row[f"{metric_key}_mean"] = mean_value
            summary_row[f"{metric_key}_std"] = population_std
            summary_row[f"{metric_key}_sample_std"] = sample_std
            summary_row[f"{metric_key}_sem"] = float(sem)
            summary_row[f"{metric_key}_ci95_half_width"] = float(ci95_half_width)
            summary_row[f"{metric_key}_ci95_low"] = float(mean_value - ci95_half_width)
            summary_row[f"{metric_key}_ci95_high"] = float(mean_value + ci95_half_width)
            summary_row[f"{metric_key}_min"] = float(values.min())
            summary_row[f"{metric_key}_max"] = float(values.max())
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: tuple(row[key_name] for key_name in group_keys))
    return summary_rows


def _run_multiseed_purified_certificate_summary(dataset, data, device):
    print("\n=== Multi-seed purified certificate summary ===")
    target_pool_rows = []
    fixed_config_per_seed_rows = []
    oracle_per_seed_rows = []
    variant_specs = _build_additional_training_variant_specs()

    for seed in MULTI_SEED_PURIFIED_SUMMARY_SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        clean_model = _build_model(dataset, device)
        clean_optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.01, weight_decay=5e-4)
        _train_model(clean_model, data, clean_optimizer)

        surrogate, adj, features, labels = train_deeprobust_surrogate(data, device=device)
        variant_bundles = [{"variant": "clean-training", "model": clean_model}]
        warm_start_states = {"clean-training": deepcopy(clean_model.state_dict())}

        for variant_spec in variant_specs:
            variant_model = _build_model(dataset, device)
            warm_start_from = variant_spec.get("warm_start_from")
            if warm_start_from is not None:
                variant_model.load_state_dict(deepcopy(warm_start_states[warm_start_from]))

            variant_optimizer = torch.optim.Adam(
                variant_model.parameters(),
                lr=variant_spec["learning_rate"],
                weight_decay=variant_spec["weight_decay"],
            )
            _train_model_with_config(
                variant_model,
                data,
                variant_optimizer,
                epochs=variant_spec["epochs"],
                train_config=deepcopy(variant_spec["train_config"]),
            )
            variant_bundles.append({"variant": variant_spec["label"], "model": variant_model})
            warm_start_states[variant_spec["label"]] = deepcopy(variant_model.state_dict())

        for bundle in variant_bundles:
            attack_pool = _collect_nettack_result_pool(
                target_model=bundle["model"],
                data=data,
                surrogate=surrogate,
                adj=adj,
                features=features,
                labels=labels,
                target_count=PURIFIED_CERTIFICATE_TARGET_COUNT,
                device=device,
            )
            target_pool_mode = "attempted-target-fallback" if attack_pool["used_fallback_pool"] else "successful-attacks"
            target_pool_rows.append(
                {
                    "seed": int(seed),
                    "model_variant": bundle["variant"],
                    "attack_variant": bundle["variant"],
                    "attempted_nodes": int(attack_pool["attempted_nodes"]),
                    "successful_attacks": int(attack_pool["successful_attacks"]),
                    "evaluated_targets": int(len(attack_pool["selected_results"])),
                    "target_count_goal": int(PURIFIED_CERTIFICATE_TARGET_COUNT),
                    "target_pool_mode": target_pool_mode,
                }
            )

            sweep_bundle = _evaluate_purified_certificate_sweep(
                model=bundle["model"],
                data=data,
                purified_certificate_results=attack_pool["selected_results"],
                variant_name=bundle["variant"],
                attack_variant=bundle["variant"],
                emit_logs=False,
            )
            fixed_config_per_seed_rows.extend({"seed": int(seed), **row} for row in sweep_bundle["candidate_summary"])
            oracle_per_seed_rows.extend({"seed": int(seed), **row} for row in sweep_bundle["oracle_summary"])

            for row in sweep_bundle["oracle_summary"]:
                print(
                    f"seed={seed} | variant={bundle['variant']} | thr={row['threshold']:.3f} | "
                    f"correct {row['correct_fraction']:.4f} | certified>0 {row['positive_certified_fraction']:.4f}"
                )

    target_pool_summary = _aggregate_multiseed_numeric_rows(
        rows=target_pool_rows,
        group_keys=["model_variant", "attack_variant", "target_pool_mode", "target_count_goal"],
        metric_keys=["attempted_nodes", "successful_attacks", "evaluated_targets"],
    )
    fixed_config_summary = _aggregate_multiseed_numeric_rows(
        rows=fixed_config_per_seed_rows,
        group_keys=["model_variant", "attack_variant", "threshold", "config_label", "mode", "p_delete", "p_add", "max_additions"],
        metric_keys=[
            "evaluated_nodes",
            "purified_plain_correct_fraction",
            "correct_fraction",
            "positive_certified_fraction",
            "max_reported_radius",
            "mean_pA_lower",
        ],
    )
    oracle_summary = _aggregate_multiseed_numeric_rows(
        rows=oracle_per_seed_rows,
        group_keys=["model_variant", "attack_variant", "threshold"],
        metric_keys=[
            "evaluated_nodes",
            "purified_plain_correct_fraction",
            "correct_fraction",
            "positive_certified_fraction",
            "mean_reported_radius",
            "max_reported_radius",
            "mean_pA_lower",
            "mean_target_edge_retention",
            "nodes_with_correct_config",
            "nodes_without_correct_config",
        ],
    )

    return {
        "target_pool_rows": target_pool_rows,
        "target_pool_summary": target_pool_summary,
        "fixed_config_per_seed_rows": fixed_config_per_seed_rows,
        "fixed_config_summary": fixed_config_summary,
        "oracle_per_seed_rows": oracle_per_seed_rows,
        "oracle_summary": oracle_summary,
    }



def run_single_experiment(
    seed: int,
    results_dir: Path,
    dataset_name: str,
    smoothing_modes,
    epochs: int,
    run_purified_multiseed_summary: bool = True,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ensure_results_dir(results_dir)
    smoothing_modes = list(smoothing_modes or DEFAULT_SMOOTHING_MODES)

    dataset = Planetoid(
        root="data/Planetoid",
        name=dataset_name,
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    model = _build_model(dataset, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    training_history, best_val, best_test = _train_model(model, data, optimizer, epochs=epochs)

    clean_metrics, _ = evaluate_with_edge_index(model, data, data.edge_index)
    print(f"\nClean test accuracy: {clean_metrics['test']:.4f}")

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

    sparse_baseline_config = SPARSE_FLIP_SWEEP[1]
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

    additional_variant_bundles = {}
    warm_start_states = {"clean-training": deepcopy(model.state_dict())}
    for variant_spec in _build_additional_training_variant_specs():
        print(f"\n=== {variant_spec['display_name']} ===")
        variant_model = _build_model(dataset, device)
        warm_start_from = variant_spec.get("warm_start_from")
        if warm_start_from is not None:
            variant_model.load_state_dict(deepcopy(warm_start_states[warm_start_from]))

        variant_optimizer = torch.optim.Adam(
            variant_model.parameters(),
            lr=variant_spec["learning_rate"],
            weight_decay=variant_spec["weight_decay"],
        )
        variant_history, variant_best_val, variant_best_test = _train_model_with_config(
            model=variant_model,
            data=data,
            optimizer=variant_optimizer,
            epochs=variant_spec["epochs"],
            train_config=deepcopy(variant_spec["train_config"]),
        )
        variant_clean_metrics, _ = evaluate_with_edge_index(variant_model, data, data.edge_index)
        variant_sparse_baseline, _, _, _ = evaluate_smoothed(
            model=variant_model,
            data=data,
            num_samples=500,
            batch_size=50,
            mode="sparse-edge-flip",
            p_delete=sparse_baseline_config["p_delete"],
            p_add=sparse_baseline_config["p_add"],
            max_additions=sparse_baseline_config["max_additions"],
        )
        _print_metrics(f"{variant_spec['label']} clean eval", variant_clean_metrics)
        _print_metrics(f"{variant_spec['label']} sparse smoothing", variant_sparse_baseline)

        variant_attack_rows = _run_variant_global_attack_evaluation(
            model=variant_model,
            data=data,
            device=device,
            variant_label=variant_spec["label"],
            heading=f"{variant_spec['display_name']} global attack evaluation",
        )
        additional_variant_bundles[variant_spec["label"]] = {
            "label": variant_spec["label"],
            "display_name": variant_spec["display_name"],
            "warm_start_from": warm_start_from or "",
            "epochs": int(variant_spec["epochs"]),
            "learning_rate": float(variant_spec["learning_rate"]),
            "train_config": deepcopy(variant_spec["train_config"]),
            "model": variant_model,
            "history": variant_history,
            "best_val": float(variant_best_val),
            "best_test": float(variant_best_test),
            "clean_metrics": variant_clean_metrics,
            "sparse_baseline": variant_sparse_baseline,
            "attack_rows": variant_attack_rows,
        }
        warm_start_states[variant_spec["label"]] = deepcopy(variant_model.state_dict())

    robust_bundle = additional_variant_bundles[MATCHED_SPARSE_TRAINING_LABEL]
    robust_model = robust_bundle["model"]
    robust_training_history = robust_bundle["history"]
    robust_best_val = robust_bundle["best_val"]
    robust_best_test = robust_bundle["best_test"]
    robust_clean_metrics = robust_bundle["clean_metrics"]
    robust_sparse_baseline = robust_bundle["sparse_baseline"]
    robust_attack_rows = robust_bundle["attack_rows"]

    certificate_bundle = additional_variant_bundles[CERTIFICATE_ORIENTED_TRAINING_LABEL]
    certificate_model = certificate_bundle["model"]
    certificate_training_history = certificate_bundle["history"]
    certificate_best_val = certificate_bundle["best_val"]
    certificate_best_test = certificate_bundle["best_test"]
    certificate_clean_metrics = certificate_bundle["clean_metrics"]
    certificate_sparse_baseline = certificate_bundle["sparse_baseline"]
    certificate_attack_rows = certificate_bundle["attack_rows"]

    print("\n=== Local target-node certified accuracy subset sweep ===")
    certificate_subset_nodes, certificate_subset_metadata = choose_correct_test_nodes(
        model,
        data,
        k=CERTIFIED_ACCURACY_NODE_COUNT,
        strategy="stratified",
        return_metadata=True,
    )
    symmetric_subset_bundle = _evaluate_symmetric_certificate_subset(
        model=model,
        data=data,
        certificate_subset_nodes=certificate_subset_nodes,
        certificate_subset_metadata=certificate_subset_metadata,
    )
    symmetric_certificate_rows = symmetric_subset_bundle["summary_rows"]
    certified_accuracy_curve_rows = symmetric_subset_bundle["curve_rows"]
    certified_accuracy_curves = symmetric_subset_bundle["curves"]
    certificate_subset_rows = symmetric_subset_bundle["subset_rows"]

    print("\n=== Sparse local target-node certified accuracy subset sweep ===")
    sparse_subset_bundle = _evaluate_sparse_certificate_subset(
        model=model,
        data=data,
        certificate_subset_nodes=certificate_subset_nodes,
        certificate_subset_metadata=certificate_subset_metadata,
    )
    sparse_certificate_accuracy_rows = sparse_subset_bundle["summary_rows"]
    sparse_certified_accuracy_curve_rows = sparse_subset_bundle["curve_rows"]
    certified_accuracy_curves.extend(sparse_subset_bundle["curves"])
    sparse_certificate_subset_rows = sparse_subset_bundle["subset_rows"]

    print("\n=== Robust-training local target-node certified accuracy subset sweep ===")
    robust_symmetric_subset_bundle = _evaluate_symmetric_certificate_subset(
        model=robust_model,
        data=data,
        certificate_subset_nodes=certificate_subset_nodes,
        certificate_subset_metadata=certificate_subset_metadata,
        curve_label_prefix="robust ",
    )
    robust_symmetric_certificate_rows = robust_symmetric_subset_bundle["summary_rows"]
    robust_certified_accuracy_curve_rows = robust_symmetric_subset_bundle["curve_rows"]
    certified_accuracy_curves.extend(robust_symmetric_subset_bundle["curves"])
    robust_certificate_subset_rows = robust_symmetric_subset_bundle["subset_rows"]

    print("\n=== Robust-training sparse local target-node certified accuracy subset sweep ===")
    robust_sparse_subset_bundle = _evaluate_sparse_certificate_subset(
        model=robust_model,
        data=data,
        certificate_subset_nodes=certificate_subset_nodes,
        certificate_subset_metadata=certificate_subset_metadata,
        curve_label_prefix="robust ",
    )
    robust_sparse_certificate_accuracy_rows = robust_sparse_subset_bundle["summary_rows"]
    robust_sparse_certified_accuracy_curve_rows = robust_sparse_subset_bundle["curve_rows"]
    certified_accuracy_curves.extend(robust_sparse_subset_bundle["curves"])
    robust_sparse_certificate_subset_rows = robust_sparse_subset_bundle["subset_rows"]

    print("\n=== Jointly-correct subset certificate comparison ===")
    joint_certificate_subset_nodes, joint_certificate_subset_metadata = choose_jointly_correct_test_nodes(
        reference_model=model,
        comparison_model=robust_model,
        data=data,
        k=CERTIFIED_ACCURACY_NODE_COUNT,
        strategy="stratified",
        return_metadata=True,
    )

    print("\n--- Joint subset: clean symmetric ---")
    joint_clean_symmetric_bundle = _evaluate_symmetric_certificate_subset(
        model=model,
        data=data,
        certificate_subset_nodes=joint_certificate_subset_nodes,
        certificate_subset_metadata=joint_certificate_subset_metadata,
    )
    print("\n--- Joint subset: clean sparse ---")
    joint_clean_sparse_bundle = _evaluate_sparse_certificate_subset(
        model=model,
        data=data,
        certificate_subset_nodes=joint_certificate_subset_nodes,
        certificate_subset_metadata=joint_certificate_subset_metadata,
    )
    print("\n--- Joint subset: robust symmetric ---")
    joint_robust_symmetric_bundle = _evaluate_symmetric_certificate_subset(
        model=robust_model,
        data=data,
        certificate_subset_nodes=joint_certificate_subset_nodes,
        certificate_subset_metadata=joint_certificate_subset_metadata,
        curve_label_prefix="joint robust ",
    )
    print("\n--- Joint subset: robust sparse ---")
    joint_robust_sparse_bundle = _evaluate_sparse_certificate_subset(
        model=robust_model,
        data=data,
        certificate_subset_nodes=joint_certificate_subset_nodes,
        certificate_subset_metadata=joint_certificate_subset_metadata,
        curve_label_prefix="joint robust ",
    )

    joint_certificate_variant_rows = []
    joint_certificate_variant_rows.extend(
        _tag_certificate_summary_rows(
            joint_clean_symmetric_bundle["summary_rows"],
            variant="clean-training",
            certificate_family="symmetric",
            subset_name="jointly-correct",
        )
    )
    joint_certificate_variant_rows.extend(
        _tag_certificate_summary_rows(
            joint_clean_sparse_bundle["summary_rows"],
            variant="clean-training",
            certificate_family="sparse",
            subset_name="jointly-correct",
        )
    )
    joint_certificate_variant_rows.extend(
        _tag_certificate_summary_rows(
            joint_robust_symmetric_bundle["summary_rows"],
            variant="matched-sparse-noisy-training",
            certificate_family="symmetric",
            subset_name="jointly-correct",
        )
    )
    joint_certificate_variant_rows.extend(
        _tag_certificate_summary_rows(
            joint_robust_sparse_bundle["summary_rows"],
            variant="matched-sparse-noisy-training",
            certificate_family="sparse",
            subset_name="jointly-correct",
        )
    )

    global_attack_rows = []
    purification_attacked_edge_index = None
    print("\n=== Global attack evaluation ===")
    for budget in GLOBAL_ATTACK_BUDGETS:
        attacked_edge_index, flipped_edges = run_prbcd_attack(model, data, budget, device)
        attacked_metrics, _ = evaluate_with_edge_index(model, data, attacked_edge_index)
        num_flips = flipped_edges.size(1) if flipped_edges.numel() > 0 else 0
        if budget == PURIFICATION_ATTACK_BUDGET:
            purification_attacked_edge_index = attacked_edge_index

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

    purification_rows = []
    if purification_attacked_edge_index is not None:
        print("\n=== Jaccard edge purification sweep ===")
        for threshold in PURIFICATION_SWEEP:
            purified_clean_edge_index, clean_purification_stats = purify_edge_index_by_jaccard(
                x=data.x,
                edge_index=data.edge_index,
                threshold=threshold,
            )
            purified_attacked_edge_index, attacked_purification_stats = purify_edge_index_by_jaccard(
                x=data.x,
                edge_index=purification_attacked_edge_index,
                threshold=threshold,
            )
            purified_clean_metrics, _ = evaluate_with_edge_index(model, data, purified_clean_edge_index)
            purified_attacked_metrics, _ = evaluate_with_edge_index(model, data, purified_attacked_edge_index)

            purification_rows.append(
                {
                    "threshold": float(threshold),
                    "attack_budget": int(PURIFICATION_ATTACK_BUDGET),
                    "clean_test_accuracy": float(purified_clean_metrics["test"]),
                    "attacked_test_accuracy": float(purified_attacked_metrics["test"]),
                    "clean_edge_retention": float(clean_purification_stats["edge_retention"]),
                    "attacked_edge_retention": float(attacked_purification_stats["edge_retention"]),
                    "clean_mean_jaccard": float(clean_purification_stats["mean_jaccard"]),
                    "attacked_mean_jaccard": float(attacked_purification_stats["mean_jaccard"]),
                }
            )

            print(
                f"threshold={threshold:.3f} | "
                f"Clean acc {purified_clean_metrics['test']:.4f} | "
                f"Attack acc {purified_attacked_metrics['test']:.4f} | "
                f"Clean keep {clean_purification_stats['edge_retention']:.3f} | "
                f"Attack keep {attacked_purification_stats['edge_retention']:.3f}"
            )

    variant_summary_rows = [
        {
            "variant": "clean-training",
            "display_name": "Clean supervised baseline",
            "warm_start_from": "",
            "training_epochs": 200,
            "learning_rate": 0.01,
            "clean_test_accuracy": float(clean_metrics["test"]),
            "best_validation_accuracy": float(best_val),
            "best_test_accuracy": float(best_test),
            "sparse_smoothed_test_accuracy": float(sparse_flip_baseline["test"]),
            "global_attack_budget_50_accuracy": float(global_attack_rows[-1]["test_accuracy"]),
        },
    ]
    for bundle in additional_variant_bundles.values():
        variant_summary_rows.append(
            {
                "variant": bundle["label"],
                "display_name": bundle["display_name"],
                "warm_start_from": bundle["warm_start_from"],
                "training_epochs": int(bundle["epochs"]),
                "learning_rate": float(bundle["learning_rate"]),
                "clean_test_accuracy": float(bundle["clean_metrics"]["test"]),
                "best_validation_accuracy": float(bundle["best_val"]),
                "best_test_accuracy": float(bundle["best_test"]),
                "sparse_smoothed_test_accuracy": float(bundle["sparse_baseline"]["test"]),
                "global_attack_budget_50_accuracy": float(bundle["attack_rows"][-1]["test_accuracy"]),
            }
        )

    nettack_logs = []
    nettack_results = []
    if NETTACK_TARGET_COUNT > 0:
        print("\n=== Nettack targeted demo ===")
        surrogate, adj, features, labels = train_deeprobust_surrogate(data, device=device)
        candidate_nodes = choose_correct_test_nodes(model, data, k=NETTACK_TARGET_COUNT, strategy="stratified")

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
    sparse_local_certificate_rows = []
    nettack_target_purification_rows = []
    nettack_target_purification_summary = []
    purified_certificate_candidate_rows = []
    purified_certificate_candidate_summary = []
    purified_certificate_rows = []
    purified_certificate_summary = []
    purified_certificate_target_pools = []
    purified_certificate_fixed_config_rows = purified_certificate_candidate_rows
    purified_certificate_fixed_config_summary = purified_certificate_candidate_summary
    purified_certificate_oracle_rows = purified_certificate_rows
    purified_certificate_oracle_summary = purified_certificate_summary
    focus_purification_rows = []
    selected_focus_summaries = []
    focus_variant_bundles = []
    focus_variant_edge_drop_rows = []
    focus_variant_sparse_flip_rows = []
    focus_variant_certificate_rows = []
    focus_variant_sparse_certificate_rows = []
    selected_focus_summary = None

    if nettack_results:
        print("\n=== Nettack target-node purification sweep ===")
        for threshold in PURIFICATION_SWEEP:
            per_threshold_rows = []
            for result in nettack_results:
                purified_edge_index, purification_stats = purify_target_node_edges_by_jaccard(
                    x=data.x,
                    edge_index=result["attacked_edge_index"],
                    target_node=result["target_node"],
                    threshold=threshold,
                )
                purified_pred, _ = predict_node(
                    model=model,
                    data=data,
                    edge_index=purified_edge_index,
                    node_idx=result["target_node"],
                )
                row = {
                    "threshold": float(threshold),
                    "target_node": int(result["target_node"]),
                    "true_label": int(result["true_label"]),
                    "attacked_pred": int(result["attacked_pred"]),
                    "purified_pred": int(purified_pred),
                    "attacked_is_correct": int(result["attacked_pred"] == result["true_label"]),
                    "purified_is_correct": int(purified_pred == result["true_label"]),
                    "recovered": int(
                        (result["attacked_pred"] != result["true_label"])
                        and (purified_pred == result["true_label"])
                    ),
                    "target_edge_retention": float(purification_stats["target_edge_retention"]),
                    "mean_target_jaccard": float(purification_stats["mean_target_jaccard"]),
                }
                nettack_target_purification_rows.append(row)
                per_threshold_rows.append(row)

            summary_row = {
                "threshold": float(threshold),
                "evaluated_nodes": int(len(per_threshold_rows)),
                "attacked_target_accuracy": (
                    sum(row["attacked_is_correct"] for row in per_threshold_rows) / len(per_threshold_rows)
                    if per_threshold_rows
                    else 0.0
                ),
                "purified_target_accuracy": (
                    sum(row["purified_is_correct"] for row in per_threshold_rows) / len(per_threshold_rows)
                    if per_threshold_rows
                    else 0.0
                ),
                "recovery_rate": (
                    sum(row["recovered"] for row in per_threshold_rows) / len(per_threshold_rows)
                    if per_threshold_rows
                    else 0.0
                ),
                "mean_target_edge_retention": (
                    sum(row["target_edge_retention"] for row in per_threshold_rows) / len(per_threshold_rows)
                    if per_threshold_rows
                    else 0.0
                ),
            }
            nettack_target_purification_summary.append(summary_row)
            print(
                f"threshold={threshold:.3f} | "
                f"Attacked target acc {summary_row['attacked_target_accuracy']:.4f} | "
                f"Purified target acc {summary_row['purified_target_accuracy']:.4f} | "
                f"Recovery rate {summary_row['recovery_rate']:.4f} | "
                f"Target keep {summary_row['mean_target_edge_retention']:.3f}"
            )

        print("\n=== Post-purification certificate sweeps ===")
        purified_certificate_models = [
            {"variant": "clean-training", "model": model, "initial_results": nettack_results},
            {"variant": MATCHED_SPARSE_TRAINING_LABEL, "model": robust_model, "initial_results": None},
            {"variant": CERTIFICATE_ORIENTED_TRAINING_LABEL, "model": certificate_model, "initial_results": None},
        ]
        for model_bundle in purified_certificate_models:
            variant_name = model_bundle["variant"]
            sweep_model = model_bundle["model"]
            attack_pool = _collect_nettack_result_pool(
                target_model=sweep_model,
                data=data,
                surrogate=surrogate,
                adj=adj,
                features=features,
                labels=labels,
                target_count=PURIFIED_CERTIFICATE_TARGET_COUNT,
                device=device,
                initial_results=model_bundle["initial_results"],
            )
            purified_certificate_results = attack_pool["selected_results"]
            target_pool_mode = "attempted-target-fallback" if attack_pool["used_fallback_pool"] else "successful-attacks"
            purified_certificate_target_pools.append(
                {
                    "model_variant": variant_name,
                    "attack_variant": variant_name,
                    "attempted_nodes": int(attack_pool["attempted_nodes"]),
                    "successful_attacks": int(attack_pool["successful_attacks"]),
                    "evaluated_targets": int(len(purified_certificate_results)),
                    "target_count_goal": int(PURIFIED_CERTIFICATE_TARGET_COUNT),
                    "target_pool_mode": target_pool_mode,
                }
            )
            print(
                f"variant={variant_name} | Using {len(purified_certificate_results)} Nettack targets "
                f"({attack_pool['successful_attacks']} successful out of {attack_pool['attempted_nodes']} attempted, "
                f"pool={target_pool_mode})"
            )
            sweep_bundle = _evaluate_purified_certificate_sweep(
                model=sweep_model,
                data=data,
                purified_certificate_results=purified_certificate_results,
                variant_name=variant_name,
                attack_variant=variant_name,
            )
            purified_certificate_candidate_rows.extend(sweep_bundle["candidate_rows"])
            purified_certificate_candidate_summary.extend(sweep_bundle["candidate_summary"])
            purified_certificate_rows.extend(sweep_bundle["oracle_rows"])
            purified_certificate_summary.extend(sweep_bundle["oracle_summary"])

            variant_focus_result, variant_focus_probe = _select_focus_result(
                sweep_model,
                data,
                attack_pool["selected_results"],
            )
            focus_summary, variant_focus_rows = _evaluate_focus_purification_case(
                model=sweep_model,
                data=data,
                focus_result=variant_focus_result,
                focus_probe=variant_focus_probe,
                variant_name=variant_name,
                attack_variant=variant_name,
                sparse_config=sparse_baseline_config,
            )
            if focus_summary is not None:
                selected_focus_summaries.append(focus_summary)
                focus_variant_bundles.append(
                    {
                        "variant_name": variant_name,
                        "attack_variant": variant_name,
                        "model": sweep_model,
                        "focus_result": variant_focus_result,
                    }
                )
                focus_purification_rows.extend(variant_focus_rows)

    if selected_focus_summaries:
        selected_focus_summary = next(
            (summary for summary in selected_focus_summaries if summary["model_variant"] == "clean-training"),
            selected_focus_summaries[0],
        )

    if focus_variant_bundles:
        for bundle in focus_variant_bundles:
            diagnostics_bundle = _evaluate_focus_legacy_diagnostics(
                model=bundle["model"],
                data=data,
                focus_result=bundle["focus_result"],
                variant_name=bundle["variant_name"],
                attack_variant=bundle["attack_variant"],
                emit_logs=True,
            )
            focus_variant_edge_drop_rows.extend(diagnostics_bundle["edge_drop_rows"])
            focus_variant_sparse_flip_rows.extend(diagnostics_bundle["sparse_flip_rows"])
            focus_variant_certificate_rows.extend(diagnostics_bundle["symmetric_certificate_rows"])
            focus_variant_sparse_certificate_rows.extend(diagnostics_bundle["sparse_certificate_rows"])

            if bundle["variant_name"] == "clean-training":
                edge_drop_rows.extend(diagnostics_bundle["edge_drop_rows"])
                sparse_flip_rows.extend(diagnostics_bundle["sparse_flip_rows"])
                certificate_rows.extend(diagnostics_bundle["symmetric_certificate_rows"])
                sparse_local_certificate_rows.extend(diagnostics_bundle["sparse_certificate_rows"])

    results_payload = {
        "seed": int(seed),
        "dataset": dataset.name,
        "smoothing_modes": smoothing_modes,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1) // 2),
        "best_validation_accuracy": float(best_val),
        "best_test_accuracy": float(best_test),
        "clean_metrics": _float_dict(clean_metrics),
        "edge_drop_baseline": _float_dict(edge_drop_baseline),
        "sparse_flip_baseline": {
            **_float_dict(sparse_flip_baseline),
            **{
                "p_delete": float(sparse_baseline_config["p_delete"]),
                "p_add": float(sparse_baseline_config["p_add"]),
                "max_additions": int(sparse_baseline_config["max_additions"]),
            },
        },
        "robust_training_config": {
            "variant": MATCHED_SPARSE_TRAINING_LABEL,
            **{
                key: (int(value) if key == "max_additions" else float(value))
                if isinstance(value, (int, float))
                else value
                for key, value in MATCHED_SPARSE_NOISE_CONFIG.items()
            },
        },
        "robust_training_clean_metrics": _float_dict(robust_clean_metrics),
        "robust_training_sparse_flip_baseline": {
            **_float_dict(robust_sparse_baseline),
            **{
                "p_delete": float(sparse_baseline_config["p_delete"]),
                "p_add": float(sparse_baseline_config["p_add"]),
                "max_additions": int(sparse_baseline_config["max_additions"]),
            },
        },
        "robust_training_best_validation_accuracy": float(robust_best_val),
        "robust_training_best_test_accuracy": float(robust_best_test),
        "robust_training_global_attack_results": robust_attack_rows,
        "certificate_oriented_training_config": {
            "variant": CERTIFICATE_ORIENTED_TRAINING_LABEL,
            "warm_start_from": "clean-training",
            "epochs": int(CERTIFICATE_ORIENTED_FINE_TUNE_EPOCHS),
            "learning_rate": float(CERTIFICATE_ORIENTED_FINE_TUNE_LR),
            **{
                key: (int(value) if key == "max_additions" else float(value))
                if isinstance(value, (int, float))
                else value
                for key, value in CERTIFICATE_ORIENTED_FINE_TUNE_CONFIG.items()
            },
        },
        "certificate_oriented_training_clean_metrics": _float_dict(certificate_clean_metrics),
        "certificate_oriented_training_sparse_flip_baseline": {
            **_float_dict(certificate_sparse_baseline),
            **{
                "p_delete": float(sparse_baseline_config["p_delete"]),
                "p_add": float(sparse_baseline_config["p_add"]),
                "max_additions": int(sparse_baseline_config["max_additions"]),
            },
        },
        "certificate_oriented_training_best_validation_accuracy": float(certificate_best_val),
        "certificate_oriented_training_best_test_accuracy": float(certificate_best_test),
        "certificate_oriented_training_global_attack_results": certificate_attack_rows,
        "training_variant_summary": variant_summary_rows,
        "purified_certificate_multiseed_seeds": MULTI_SEED_PURIFIED_SUMMARY_SEEDS,
        "purified_certificate_multiseed_confidence_level": 0.95,
        "jaccard_purification_attack_budget": int(PURIFICATION_ATTACK_BUDGET),
        "jaccard_purification_sweep": purification_rows,
        "local_certificate_accuracy_subset_nodes": certificate_subset_nodes,
        "local_certificate_accuracy_subset_metadata": list(certificate_subset_metadata.values()),
        "joint_certificate_subset_nodes": joint_certificate_subset_nodes,
        "joint_certificate_subset_metadata": list(joint_certificate_subset_metadata.values()),
        "joint_certificate_variant_summary": joint_certificate_variant_rows,
        "nettack_target_purification_summary": nettack_target_purification_summary,
        "nettack_target_purification_rows": nettack_target_purification_rows,
        "purified_certificate_target_pools": purified_certificate_target_pools,
        "purified_certificate_fixed_config_summary": purified_certificate_fixed_config_summary,
        "purified_certificate_fixed_config_rows": purified_certificate_fixed_config_rows,
        "purified_certificate_oracle_summary": purified_certificate_oracle_summary,
        "purified_certificate_oracle_rows": purified_certificate_oracle_rows,
        "purified_certificate_candidate_summary": purified_certificate_fixed_config_summary,
        "purified_certificate_candidate_rows": purified_certificate_fixed_config_rows,
        "purified_certificate_summary": purified_certificate_oracle_summary,
        "purified_certificate_rows": purified_certificate_oracle_rows,
        "focus_purification_certificate_sweep": focus_purification_rows,
        "focus_variant_edge_drop_sweep": focus_variant_edge_drop_rows,
        "focus_variant_sparse_edge_flip_sweep": focus_variant_sparse_flip_rows,
        "focus_variant_local_certificate_sweep": focus_variant_certificate_rows,
        "focus_variant_sparse_local_certificate_sweep": focus_variant_sparse_certificate_rows,
        "local_certificate_accuracy_subset_sweep": symmetric_certificate_rows,
        "local_certified_accuracy_curve": certified_accuracy_curve_rows,
        "local_certificate_subset_rows": certificate_subset_rows,
        "sparse_local_certificate_accuracy_subset_sweep": sparse_certificate_accuracy_rows,
        "sparse_local_certified_accuracy_curve": sparse_certified_accuracy_curve_rows,
        "sparse_local_certificate_subset_rows": sparse_certificate_subset_rows,
        "robust_local_certificate_accuracy_subset_sweep": robust_symmetric_certificate_rows,
        "robust_local_certified_accuracy_curve": robust_certified_accuracy_curve_rows,
        "robust_local_certificate_subset_rows": robust_certificate_subset_rows,
        "robust_sparse_local_certificate_accuracy_subset_sweep": robust_sparse_certificate_accuracy_rows,
        "robust_sparse_local_certified_accuracy_curve": robust_sparse_certified_accuracy_curve_rows,
        "robust_sparse_local_certificate_subset_rows": robust_sparse_certificate_subset_rows,
        "global_attack_results": global_attack_rows,
        "nettack_examples": nettack_logs,
        "selected_focus_node": selected_focus_summary,
        "edge_drop_sweep": edge_drop_rows,
        "sparse_edge_flip_sweep": sparse_flip_rows,
        "local_certificate_sweep": certificate_rows,
        "sparse_local_certificate_sweep": sparse_local_certificate_rows,
        "artifacts_dir": str(results_dir),
    }

    symmetric_degree_bucket_rows = _summarize_certificate_rows(
        rows=certificate_subset_rows,
        config_keys=["p_flip"],
        group_keys=["degree_bucket"],
    )
    symmetric_margin_bucket_rows = _summarize_certificate_rows(
        rows=certificate_subset_rows,
        config_keys=["p_flip"],
        group_keys=["margin_bucket"],
    )
    sparse_degree_bucket_rows = _summarize_certificate_rows(
        rows=sparse_certificate_subset_rows,
        config_keys=["p_delete", "p_add", "max_additions"],
        group_keys=["degree_bucket"],
    )
    sparse_margin_bucket_rows = _summarize_certificate_rows(
        rows=sparse_certificate_subset_rows,
        config_keys=["p_delete", "p_add", "max_additions"],
        group_keys=["margin_bucket"],
    )
    robust_symmetric_degree_bucket_rows = _summarize_certificate_rows(
        rows=robust_certificate_subset_rows,
        config_keys=["p_flip"],
        group_keys=["degree_bucket"],
    )
    robust_symmetric_margin_bucket_rows = _summarize_certificate_rows(
        rows=robust_certificate_subset_rows,
        config_keys=["p_flip"],
        group_keys=["margin_bucket"],
    )
    robust_sparse_degree_bucket_rows = _summarize_certificate_rows(
        rows=robust_sparse_certificate_subset_rows,
        config_keys=["p_delete", "p_add", "max_additions"],
        group_keys=["degree_bucket"],
    )
    robust_sparse_margin_bucket_rows = _summarize_certificate_rows(
        rows=robust_sparse_certificate_subset_rows,
        config_keys=["p_delete", "p_add", "max_additions"],
        group_keys=["margin_bucket"],
    )
    results_payload.update(
        {
            "selected_focus_nodes": selected_focus_summaries,
            "symmetric_degree_bucket_summary": symmetric_degree_bucket_rows,
            "symmetric_margin_bucket_summary": symmetric_margin_bucket_rows,
            "sparse_degree_bucket_summary": sparse_degree_bucket_rows,
            "sparse_margin_bucket_summary": sparse_margin_bucket_rows,
            "robust_symmetric_degree_bucket_summary": robust_symmetric_degree_bucket_rows,
            "robust_symmetric_margin_bucket_summary": robust_symmetric_margin_bucket_rows,
            "robust_sparse_degree_bucket_summary": robust_sparse_degree_bucket_rows,
            "robust_sparse_margin_bucket_summary": robust_sparse_margin_bucket_rows,
        }
    )

    multiseed_purified_bundle = None
    if run_purified_multiseed_summary:
        multiseed_purified_bundle = _run_multiseed_purified_certificate_summary(dataset, data, device)
        results_payload.update(
            {
                "purified_certificate_multiseed_target_pools": multiseed_purified_bundle["target_pool_rows"],
                "purified_certificate_multiseed_target_pool_summary": multiseed_purified_bundle["target_pool_summary"],
                "purified_certificate_multiseed_fixed_config_per_seed": multiseed_purified_bundle["fixed_config_per_seed_rows"],
                "purified_certificate_multiseed_fixed_config_summary": multiseed_purified_bundle["fixed_config_summary"],
                "purified_certificate_multiseed_oracle_per_seed": multiseed_purified_bundle["oracle_per_seed_rows"],
                "purified_certificate_multiseed_oracle_summary": multiseed_purified_bundle["oracle_summary"],
            }
        )

    save_csv_rows(results_dir / "training_history.csv", training_history)
    save_csv_rows(results_dir / "robust_training_history.csv", robust_training_history)
    save_csv_rows(results_dir / "certificate_oriented_training_history.csv", certificate_training_history)
    save_csv_rows(results_dir / "training_variant_summary.csv", variant_summary_rows)
    save_csv_rows(results_dir / "robust_attack_budget_accuracy.csv", robust_attack_rows)
    save_csv_rows(results_dir / "certificate_oriented_attack_budget_accuracy.csv", certificate_attack_rows)
    save_csv_rows(results_dir / "symmetric_certificate_sweep.csv", symmetric_certificate_rows)
    save_csv_rows(results_dir / "certified_accuracy_curve.csv", certified_accuracy_curve_rows)
    save_csv_rows(results_dir / "certificate_subset_rows.csv", certificate_subset_rows)
    save_csv_rows(results_dir / "symmetric_degree_bucket_summary.csv", symmetric_degree_bucket_rows)
    save_csv_rows(results_dir / "symmetric_margin_bucket_summary.csv", symmetric_margin_bucket_rows)
    save_csv_rows(results_dir / "sparse_certificate_subset_sweep.csv", sparse_certificate_accuracy_rows)
    save_csv_rows(results_dir / "sparse_certified_accuracy_curve.csv", sparse_certified_accuracy_curve_rows)
    save_csv_rows(results_dir / "sparse_certificate_subset_rows.csv", sparse_certificate_subset_rows)
    save_csv_rows(results_dir / "sparse_degree_bucket_summary.csv", sparse_degree_bucket_rows)
    save_csv_rows(results_dir / "sparse_margin_bucket_summary.csv", sparse_margin_bucket_rows)
    save_csv_rows(results_dir / "robust_symmetric_certificate_sweep.csv", robust_symmetric_certificate_rows)
    save_csv_rows(results_dir / "robust_certified_accuracy_curve.csv", robust_certified_accuracy_curve_rows)
    save_csv_rows(results_dir / "robust_certificate_subset_rows.csv", robust_certificate_subset_rows)
    save_csv_rows(results_dir / "robust_symmetric_degree_bucket_summary.csv", robust_symmetric_degree_bucket_rows)
    save_csv_rows(results_dir / "robust_symmetric_margin_bucket_summary.csv", robust_symmetric_margin_bucket_rows)
    save_csv_rows(results_dir / "robust_sparse_certificate_subset_sweep.csv", robust_sparse_certificate_accuracy_rows)
    save_csv_rows(results_dir / "robust_sparse_certified_accuracy_curve.csv", robust_sparse_certified_accuracy_curve_rows)
    save_csv_rows(results_dir / "robust_sparse_certificate_subset_rows.csv", robust_sparse_certificate_subset_rows)
    save_csv_rows(results_dir / "robust_sparse_degree_bucket_summary.csv", robust_sparse_degree_bucket_rows)
    save_csv_rows(results_dir / "robust_sparse_margin_bucket_summary.csv", robust_sparse_margin_bucket_rows)
    save_csv_rows(results_dir / "joint_certificate_variant_summary.csv", joint_certificate_variant_rows)
    save_csv_rows(results_dir / "purification_sweep.csv", purification_rows)
    save_csv_rows(results_dir / "nettack_target_purification_sweep.csv", nettack_target_purification_summary)
    save_csv_rows(results_dir / "nettack_target_purification_rows.csv", nettack_target_purification_rows)
    save_csv_rows(results_dir / "purified_certificate_target_pools.csv", purified_certificate_target_pools)
    save_csv_rows(results_dir / "purified_certificate_fixed_config_sweep.csv", purified_certificate_fixed_config_summary)
    save_csv_rows(results_dir / "purified_certificate_fixed_config_rows.csv", purified_certificate_fixed_config_rows)
    save_csv_rows(results_dir / "purified_certificate_oracle_sweep.csv", purified_certificate_oracle_summary)
    save_csv_rows(results_dir / "purified_certificate_oracle_rows.csv", purified_certificate_oracle_rows)
    save_csv_rows(results_dir / "purified_certificate_candidate_sweep.csv", purified_certificate_fixed_config_summary)
    save_csv_rows(results_dir / "purified_certificate_candidate_rows.csv", purified_certificate_fixed_config_rows)
    save_csv_rows(results_dir / "purified_certificate_sweep.csv", purified_certificate_oracle_summary)
    save_csv_rows(results_dir / "purified_certificate_rows.csv", purified_certificate_oracle_rows)
    save_csv_rows(results_dir / "focus_case_study_summary.csv", selected_focus_summaries)
    save_csv_rows(results_dir / "focus_purification_certificate_sweep.csv", focus_purification_rows)
    save_csv_rows(results_dir / "focus_variant_edge_drop_sweep.csv", focus_variant_edge_drop_rows)
    save_csv_rows(results_dir / "focus_variant_sparse_edge_flip_sweep.csv", focus_variant_sparse_flip_rows)
    save_csv_rows(results_dir / "focus_variant_local_certificate_sweep.csv", focus_variant_certificate_rows)
    save_csv_rows(results_dir / "focus_variant_sparse_local_certificate_sweep.csv", focus_variant_sparse_certificate_rows)
    if multiseed_purified_bundle is not None:
        save_csv_rows(results_dir / "purified_certificate_multiseed_target_pools.csv", multiseed_purified_bundle["target_pool_rows"])
        save_csv_rows(results_dir / "purified_certificate_multiseed_target_pool_summary.csv", multiseed_purified_bundle["target_pool_summary"])
        save_csv_rows(results_dir / "purified_certificate_multiseed_fixed_config_per_seed.csv", multiseed_purified_bundle["fixed_config_per_seed_rows"])
        save_csv_rows(results_dir / "purified_certificate_multiseed_fixed_config_summary.csv", multiseed_purified_bundle["fixed_config_summary"])
        save_csv_rows(results_dir / "purified_certificate_multiseed_oracle_per_seed.csv", multiseed_purified_bundle["oracle_per_seed_rows"])
        save_csv_rows(results_dir / "purified_certificate_multiseed_oracle_summary.csv", multiseed_purified_bundle["oracle_summary"])
    save_csv_rows(results_dir / "attack_budget_accuracy.csv", global_attack_rows)
    save_csv_rows(results_dir / "edge_drop_sweep.csv", edge_drop_rows)
    save_csv_rows(results_dir / "sparse_edge_flip_sweep.csv", sparse_flip_rows)
    save_csv_rows(results_dir / "local_certificate_sweep.csv", certificate_rows)
    save_csv_rows(results_dir / "sparse_local_certificate_sweep.csv", sparse_local_certificate_rows)
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
    plot_dual_accuracy_curve(
        output_path=results_dir / "purification_tradeoff.png",
        rows=purification_rows,
        x_key="threshold",
        clean_key="clean_test_accuracy",
        attacked_key="attacked_test_accuracy",
        title="Jaccard edge purification tradeoff",
        x_label="Jaccard threshold",
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
            run_purified_multiseed_summary=(len(seeds) == 1),
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
