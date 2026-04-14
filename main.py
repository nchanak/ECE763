import random

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.attack import run_prbcd_attack
from src.models import GCN
from src.nettack import choose_correct_test_nodes, run_nettack_on_node, train_deeprobust_surrogate
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
LOCAL_CERTIFICATE_SAMPLES = 400
CERTIFICATE_MAX_RADIUS = 5
FOCUS_SELECTION_P_FLIP = 0.02
FOCUS_SELECTION_SAMPLES = 100


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _float_dict(metrics):
    return {key: float(value) for key, value in metrics.items()}



def _print_metrics(prefix, metrics):
    print(
        f"{prefix} | "
        f"Train {metrics['train']:.4f} | "
        f"Val {metrics['val']:.4f} | "
        f"Test {metrics['test']:.4f}"
    )



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



def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = ensure_results_dir()

    dataset = Planetoid(
        root="data/Planetoid",
        name="Cora",
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
    training_history, best_val, best_test = _train_model(model, data, optimizer)

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

    print("\n=== Local target-node certified accuracy subset sweep ===")
    certificate_subset_nodes = choose_correct_test_nodes(model, data, k=CERTIFIED_ACCURACY_NODE_COUNT)
    symmetric_certificate_rows = []
    certified_accuracy_curve_rows = []
    certified_accuracy_curves = []
    certificate_subset_rows = []

    for p_flip in CERTIFIED_ACCURACY_SWEEP:
        per_node_certificates = []
        for node_idx in certificate_subset_nodes:
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
        symmetric_certificate_rows.append(row)

        curve_rows = []
        for point in certified_accuracy_curve:
            curve_row = {
                "p_flip": float(p_flip),
                "radius": int(point["radius"]),
                "certified_accuracy": float(point["certified_accuracy"]),
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
            f"Subset accuracy {row['correct_fraction']:.4f} | "
            f"Certified acc @ r>=1 {row['positive_certified_accuracy']:.4f} | "
            f"Mean radius on correct {row['mean_certified_radius_on_correct']:.3f}"
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

    print("\n=== Nettack targeted demo ===")
    surrogate, adj, features, labels = train_deeprobust_surrogate(data, device=device)
    candidate_nodes = choose_correct_test_nodes(model, data, k=5)
    nettack_logs = []
    nettack_results = []

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
            f"\nSelected target node {node_idx} for the certificate case study | "
            f"Clean probe correct: {focus_probe['clean_probe']['is_correct']} | "
            f"Attack probe correct: {focus_probe['attack_probe']['is_correct']}"
        )

        print("\n=== Edge-drop smoothing sweep on clean vs Nettack graph ===")
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

        print("\n=== Sparse edge-flip smoothing sweep ===")
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

        print("\n=== Target-node symmetric edge-flip certificate sweep ===")
        for p_flip in LOCAL_CERTIFICATE_SWEEP:
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

    results_payload = {
        "dataset": dataset.name,
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
        "local_certificate_accuracy_subset_nodes": certificate_subset_nodes,
        "local_certificate_accuracy_subset_sweep": symmetric_certificate_rows,
        "local_certified_accuracy_curve": certified_accuracy_curve_rows,
        "local_certificate_subset_rows": certificate_subset_rows,
        "global_attack_results": global_attack_rows,
        "nettack_examples": nettack_logs,
        "selected_focus_node": selected_focus_summary,
        "edge_drop_sweep": edge_drop_rows,
        "sparse_edge_flip_sweep": sparse_flip_rows,
        "local_certificate_sweep": certificate_rows,
        "artifacts_dir": str(results_dir),
    }

    save_csv_rows(results_dir / "training_history.csv", training_history)
    save_csv_rows(results_dir / "symmetric_certificate_sweep.csv", symmetric_certificate_rows)
    save_csv_rows(results_dir / "certified_accuracy_curve.csv", certified_accuracy_curve_rows)
    save_csv_rows(results_dir / "certificate_subset_rows.csv", certificate_subset_rows)
    save_csv_rows(results_dir / "attack_budget_accuracy.csv", global_attack_rows)
    save_csv_rows(results_dir / "edge_drop_sweep.csv", edge_drop_rows)
    save_csv_rows(results_dir / "sparse_edge_flip_sweep.csv", sparse_flip_rows)
    save_csv_rows(results_dir / "local_certificate_sweep.csv", certificate_rows)
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
    plot_certificate_radius(
        output_path=results_dir / "local_certificate_radius.png",
        rows=certificate_rows,
    )

    print(f"\nSaved result artifacts to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
