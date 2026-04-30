import argparse
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import main
from src.nettack import train_deeprobust_surrogate
from src.purification import purify_target_node_edges_by_jaccard
from src.smoothing import (
    certify_asymmetric_budget_from_bounds,
    certify_asymmetric_radius_from_bounds,
    target_node_pair_counts,
)
from src.train import evaluate_smoothed_node_with_edge_index


UNIT_BUDGETS = [(1, 0), (0, 1), (1, 1)]


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_experiment_configs():
    configs = []
    for family, sweep in [
        ("legacy-low-noise", main.LEGACY_SPARSE_CERTIFICATE_CANDIDATE_SWEEP),
        ("active-cert-grid", main.SPARSE_CERTIFICATE_SWEEP),
    ]:
        for config in sweep:
            if config["mode"] != "sparse-edge-flip":
                continue
            configs.append({**config, "config_family": family})
    return configs


def _summarize_by_key(rows, key_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)

    summary_rows = []
    for key, group_rows in grouped.items():
        group = dict(zip(key_fields, key))
        correct_rows = [row for row in group_rows if row["is_correct"]]
        correct_count = len(correct_rows)
        summary_rows.append(
            {
                **group,
                "evaluated_targets": int(len(group_rows)),
                "correct_targets": int(correct_count),
                "positive_fraction_on_correct": (
                    sum(row["reported_certified_radius"] > 0 for row in correct_rows) / correct_count
                    if correct_count > 0
                    else 0.0
                ),
                "strict_positive_fraction_on_correct": (
                    sum(row["strict_total_radius"] > 0 for row in correct_rows) / correct_count
                    if correct_count > 0
                    else 0.0
                ),
                "runner_positive_fraction_on_correct": (
                    sum(row["runner_total_radius"] > 0 for row in correct_rows) / correct_count
                    if correct_count > 0
                    else 0.0
                ),
                "runner_relief_fraction_on_correct": (
                    sum(row["runner_total_radius"] > row["strict_total_radius"] for row in correct_rows) / correct_count
                    if correct_count > 0
                    else 0.0
                ),
                "unit_delete_feasible_fraction_on_correct": (
                    sum(row["strict_unit_delete_feasible"] for row in correct_rows) / correct_count
                    if correct_count > 0
                    else 0.0
                ),
                "unit_add_feasible_fraction_on_correct": (
                    sum(row["strict_unit_add_feasible"] for row in correct_rows) / correct_count
                    if correct_count > 0
                    else 0.0
                ),
                "budget_search_mismatches": int(
                    sum(not row["budget_search_matches_raw_radius"] for row in group_rows)
                ),
                "mean_pA_lower_on_correct": (
                    sum(row["pA_lower"] for row in correct_rows) / correct_count if correct_count > 0 else 0.0
                ),
                "mean_p_rest_upper_on_correct": (
                    sum(row["p_rest_upper"] for row in correct_rows) / correct_count if correct_count > 0 else 0.0
                ),
            }
        )
    return summary_rows


def main_entry():
    parser = argparse.ArgumentParser(description="Run the asymmetric certificate isolation experiment")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--train-epochs", type=int, default=120)
    parser.add_argument("--target-count", type=int, default=6)
    parser.add_argument("--selection-samples", type=int, default=60)
    parser.add_argument("--certification-samples", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="results/asymmetric_certificate_isolation_v1",
        help="Directory for detailed and summary CSV outputs",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    main.set_seed(args.seed)

    dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    model = main._build_model(dataset, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    main._train_model(model, data, optimizer, epochs=args.train_epochs)

    surrogate, adj, features, labels = train_deeprobust_surrogate(data, device=device)
    attack_pool = main._collect_nettack_result_pool(
        target_model=model,
        data=data,
        surrogate=surrogate,
        adj=adj,
        features=features,
        labels=labels,
        target_count=args.target_count,
        device=device,
    )
    configs = _build_experiment_configs()

    detail_rows = []
    for result in attack_pool["selected_results"]:
        target_node = int(result["target_node"])
        purified_edge_index, purification_stats = purify_target_node_edges_by_jaccard(
            x=data.x,
            edge_index=result["attacked_edge_index"],
            target_node=target_node,
            threshold=args.threshold,
        )
        num_present, num_absent = target_node_pair_counts(
            edge_index=purified_edge_index,
            num_nodes=data.num_nodes,
            target_node=target_node,
        )

        for config in configs:
            certificate, _, smoothed_pred = evaluate_smoothed_node_with_edge_index(
                model=model,
                data=data,
                edge_index=purified_edge_index,
                node_idx=target_node,
                num_samples=args.certification_samples,
                selection_num_samples=args.selection_samples,
                certification_num_samples=args.certification_samples,
                mode=config["mode"],
                p_delete=config["p_delete"],
                p_add=config["p_add"],
                max_additions=config["max_additions"],
                certificate_p_delete=config["p_delete"],
                certificate_p_add=config["p_add"],
                certificate_alpha=main.SMOOTHING_CERTIFICATE_ALPHA,
                certificate_max_radius=main.CERTIFICATE_MAX_RADIUS,
                certificate_max_delete=main.ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                certificate_max_add=main.ASYMMETRIC_CERTIFICATE_MAX_ADD,
            )

            strict_upper = float(certificate["p_rest_upper"])
            runner_upper = min(float(certificate["pB_upper"]), strict_upper)
            strict_search = certify_asymmetric_radius_from_bounds(
                p_lower=float(certificate["pA_lower"]),
                p_upper=strict_upper,
                p_delete=config["p_delete"],
                p_add=config["p_add"],
                num_present=num_present,
                num_absent=num_absent,
                max_radius=main.CERTIFICATE_MAX_RADIUS,
                max_delete=main.ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                max_add=main.ASYMMETRIC_CERTIFICATE_MAX_ADD,
            )
            runner_search = certify_asymmetric_radius_from_bounds(
                p_lower=float(certificate["pA_lower"]),
                p_upper=runner_upper,
                p_delete=config["p_delete"],
                p_add=config["p_add"],
                num_present=num_present,
                num_absent=num_absent,
                max_radius=main.CERTIFICATE_MAX_RADIUS,
                max_delete=main.ASYMMETRIC_CERTIFICATE_MAX_DELETE,
                max_add=main.ASYMMETRIC_CERTIFICATE_MAX_ADD,
            )
            unit_budget_results = {}
            for delete_budget, add_budget in UNIT_BUDGETS:
                key = f"{delete_budget}_{add_budget}"
                unit_budget_results[f"strict_budget_{key}_feasible"] = int(
                    certify_asymmetric_budget_from_bounds(
                        p_lower=float(certificate["pA_lower"]),
                        p_upper=strict_upper,
                        p_delete=config["p_delete"],
                        p_add=config["p_add"],
                        delete_budget=delete_budget,
                        add_budget=add_budget,
                    )
                )
                unit_budget_results[f"runner_budget_{key}_feasible"] = int(
                    certify_asymmetric_budget_from_bounds(
                        p_lower=float(certificate["pA_lower"]),
                        p_upper=runner_upper,
                        p_delete=config["p_delete"],
                        p_add=config["p_add"],
                        delete_budget=delete_budget,
                        add_budget=add_budget,
                    )
                )

            detail_rows.append(
                {
                    "seed": int(args.seed),
                    "threshold": float(args.threshold),
                    "config_family": config["config_family"],
                    "config_label": config["label"],
                    "p_delete": float(config["p_delete"]),
                    "p_add": float(config["p_add"]),
                    "max_additions": int(config["max_additions"]),
                    "target_node": target_node,
                    "true_label": int(result["true_label"]),
                    "smoothed_pred": int(smoothed_pred),
                    "is_correct": int(certificate["is_correct"]),
                    "reported_certified_radius": int(certificate["reported_certified_radius"]),
                    "raw_asymmetric_total_radius": int(certificate["raw_asymmetric_total_radius"]),
                    "strict_total_radius": int(strict_search["total_radius"]),
                    "runner_total_radius": int(runner_search["total_radius"]),
                    "strict_max_delete_budget": int(strict_search["max_delete_budget"]),
                    "strict_max_add_budget": int(strict_search["max_add_budget"]),
                    "runner_max_delete_budget": int(runner_search["max_delete_budget"]),
                    "runner_max_add_budget": int(runner_search["max_add_budget"]),
                    "budget_search_matches_raw_radius": int(
                        int(strict_search["total_radius"]) == int(certificate["raw_asymmetric_total_radius"])
                    ),
                    "pA_lower": float(certificate["pA_lower"]),
                    "pB_upper": float(certificate["pB_upper"]),
                    "p_rest_upper": strict_upper,
                    "num_present": int(num_present),
                    "num_absent": int(num_absent),
                    "target_edge_retention": float(purification_stats["target_edge_retention"]),
                    "strict_unit_delete_feasible": int(unit_budget_results["strict_budget_1_0_feasible"]),
                    "strict_unit_add_feasible": int(unit_budget_results["strict_budget_0_1_feasible"]),
                    "strict_unit_joint_feasible": int(unit_budget_results["strict_budget_1_1_feasible"]),
                    "runner_unit_delete_feasible": int(unit_budget_results["runner_budget_1_0_feasible"]),
                    "runner_unit_add_feasible": int(unit_budget_results["runner_budget_0_1_feasible"]),
                    "runner_unit_joint_feasible": int(unit_budget_results["runner_budget_1_1_feasible"]),
                }
            )

    config_summary_rows = _summarize_by_key(detail_rows, ["config_family", "config_label", "p_delete", "p_add"])
    family_summary_rows = _summarize_by_key(detail_rows, ["config_family"])

    output_dir = main.ensure_results_dir(args.output_dir)
    main.save_csv_rows(output_dir / "asymmetric_certificate_isolation_detail.csv", detail_rows)
    main.save_csv_rows(output_dir / "asymmetric_certificate_isolation_config_summary.csv", config_summary_rows)
    main.save_csv_rows(output_dir / "asymmetric_certificate_isolation_family_summary.csv", family_summary_rows)

    print({"saved_to": str(output_dir.resolve())})
    for row in family_summary_rows:
        print(
            f"family={row['config_family']} | correct={row['correct_targets']} | "
            f"strict>0={row['strict_positive_fraction_on_correct']:.4f} | "
            f"runner-relief={row['runner_relief_fraction_on_correct']:.4f} | "
            f"budget-mismatches={row['budget_search_mismatches']}"
        )


if __name__ == "__main__":
    main_entry()