import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt


def ensure_results_dir(base_dir: str = "results") -> Path:
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_serializable(value: Any):
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def save_json_report(output_path: Path, payload: Dict[str, Any]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2)


def save_csv_rows(output_path: Path, rows: Sequence[Dict[str, Any]]):
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_serializable(value) for key, value in row.items()})


def plot_attack_budget_accuracy(output_path: Path, rows: Sequence[Dict[str, Any]], clean_accuracy: Optional[float] = None):
    if not rows:
        return

    budgets = [row["budget"] for row in rows]
    accuracies = [row["test_accuracy"] for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(budgets, accuracies, marker="o", linewidth=2, label="PRBCD attacked")
    if clean_accuracy is not None:
        plt.axhline(clean_accuracy, color="black", linestyle="--", linewidth=1.5, label="Clean")
    plt.xlabel("Perturbation budget")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy Under Global Structural Attack")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_dual_accuracy_curve(
    output_path: Path,
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    clean_key: str,
    attacked_key: str,
    title: str,
    x_label: str,
):
    if not rows:
        return

    x_values = [row[x_key] for row in rows]
    clean_values = [row[clean_key] for row in rows]
    attacked_values = [row[attacked_key] for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(x_values, clean_values, marker="o", linewidth=2, label="Clean graph")
    plt.plot(x_values, attacked_values, marker="s", linewidth=2, label="Attacked graph")
    plt.xlabel(x_label)
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_certificate_radius(output_path: Path, rows: Sequence[Dict[str, Any]]):
    if not rows:
        return

    flip_rates = [row["p_flip"] for row in rows]
    clean_radius = [row.get("clean_reported_certified_radius", row["clean_certified_radius"]) for row in rows]
    attacked_radius = [row.get("attacked_reported_certified_radius", row["attacked_certified_radius"]) for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(flip_rates, clean_radius, marker="o", linewidth=2, label="Clean target node")
    plt.plot(flip_rates, attacked_radius, marker="s", linewidth=2, label="Attacked target node")
    plt.xlabel("Symmetric local edge-flip probability")
    plt.ylabel("Certified perturbation size")
    plt.title("Target-Node Certificate Sweep")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_certified_accuracy_curves(output_path: Path, curves: Sequence[Dict[str, Any]]):
    if not curves:
        return

    plt.figure(figsize=(7, 4))
    for curve in curves:
        rows = curve.get("rows", [])
        if not rows:
            continue
        x_values = [row["radius"] for row in rows]
        y_values = [row["certified_accuracy"] for row in rows]
        label = curve.get("label", "certificate")
        plt.plot(x_values, y_values, marker="o", linewidth=2, label=label)

    plt.xlabel("Certified perturbation size")
    plt.ylabel("Certified accuracy")
    plt.title("Certified Accuracy on Target-Node Subset")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
