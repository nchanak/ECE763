from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_WINNER_ONLY_COMBOS = [
    (dataset, architecture, attack_mode)
    for dataset in ("Cora", "CiteSeer", "PubMed")
    for architecture in ("gcn", "graphsage")
    for attack_mode in ("standard", "adaptive-purified")
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return 0.0
    return float(value)


def _to_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return 0
    return int(float(value))


def _group_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def _pick_best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(
        rows,
        key=lambda row: (
            _to_float(row, "positive_certified_fraction_mean"),
            _to_float(row, "correct_fraction_mean"),
            _to_float(row, "max_reported_radius_mean"),
            -_to_float(row, "abstained_fraction_mean"),
        ),
    )


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a final markdown report from aggregate result CSVs.")
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("results") / "training_variant_summary.csv",
    )
    parser.add_argument(
        "--mainline-dir",
        type=Path,
        default=Path("results") / "purified_mainline_multiseed_v4",
    )
    parser.add_argument(
        "--winner-only-dir",
        type=Path,
        default=Path("results") / "winner_only_benchmark_suite_v1",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results") / "final_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_rows = _read_csv_rows(args.training_summary)
    mainline_fixed_rows = _read_csv_rows(args.mainline_dir / "purified_certificate_multiseed_fixed_config_summary.csv")
    mainline_oracle_rows = _read_csv_rows(args.mainline_dir / "purified_certificate_multiseed_oracle_summary.csv")
    winner_fixed_rows = _read_csv_rows(args.winner_only_dir / "winner_only_fixed_config_summary.csv")
    winner_selector_rows = _read_csv_rows(args.winner_only_dir / "winner_only_selector_summary.csv")
    winner_oracle_rows = _read_csv_rows(args.winner_only_dir / "winner_only_oracle_summary.csv")
    winner_pool_rows = _read_csv_rows(args.winner_only_dir / "winner_only_target_pool_summary.csv")

    mainline_fixed_best = {
        variant_key[0]: _pick_best_row(rows)
        for variant_key, rows in _group_rows(mainline_fixed_rows, ("model_variant",)).items()
    }
    mainline_oracle_best = {
        variant_key[0]: _pick_best_row(rows)
        for variant_key, rows in _group_rows(mainline_oracle_rows, ("model_variant",)).items()
    }

    winner_fixed_best = {
        combo: _pick_best_row(rows)
        for combo, rows in _group_rows(winner_fixed_rows, ("dataset", "model_architecture", "attack_mode")).items()
    }
    winner_selector_best = {
        combo: _pick_best_row(rows)
        for combo, rows in _group_rows(winner_selector_rows, ("dataset", "model_architecture", "attack_mode")).items()
    }
    winner_oracle_best = {
        combo: _pick_best_row(rows)
        for combo, rows in _group_rows(winner_oracle_rows, ("dataset", "model_architecture", "attack_mode")).items()
    }

    completed_combos = set(winner_fixed_best)
    missing_combos = [combo for combo in EXPECTED_WINNER_ONLY_COMBOS if combo not in completed_combos]
    datasets_present = sorted({combo[0] for combo in completed_combos})

    figure_dir = args.winner_only_dir / "figures"
    figure_paths = [
        figure_dir / "winner_only_best_fixed_overview.png",
        figure_dir / "winner_only_selector_oracle_overview.png",
        figure_dir / "winner_only_target_pool_coverage.png",
    ]

    lines: list[str] = []
    lines.append("# Final Report")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- Winner-only completed combinations: {len(completed_combos)} / {len(EXPECTED_WINNER_ONLY_COMBOS)}")
    lines.append(f"- Winner-only datasets present: {', '.join(datasets_present) if datasets_present else 'none'}")
    if missing_combos:
        lines.append("- Missing winner-only combinations:")
        for dataset, architecture, attack_mode in missing_combos:
            lines.append(f"  - {dataset} / {architecture} / {attack_mode}")
    else:
        lines.append("- Winner-only cross-dataset coverage is complete.")
    lines.append("")

    lines.append("## Training Baselines")
    lines.append("")
    lines.append("| Variant | Clean Test | Best Val | Global Attack @ 50 |")
    lines.append("|---|---:|---:|---:|")
    for row in training_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    _format_num(_to_float(row, "clean_test_accuracy")),
                    _format_num(_to_float(row, "best_validation_accuracy")),
                    _format_num(_to_float(row, "global_attack_budget_50_accuracy")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Cora Mainline Best Results")
    lines.append("")
    lines.append("| Variant | Best Fixed | Fixed Thr | Fixed Pos Cert | Fixed Correct | Best Oracle Thr | Oracle Pos Cert | Oracle Correct |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for variant in sorted(mainline_fixed_best):
        fixed = mainline_fixed_best[variant]
        oracle = mainline_oracle_best.get(variant, fixed)
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    fixed["config_label"],
                    fixed["threshold"],
                    _format_pct(_to_float(fixed, "positive_certified_fraction_mean")),
                    _format_pct(_to_float(fixed, "correct_fraction_mean")),
                    oracle["threshold"],
                    _format_pct(_to_float(oracle, "positive_certified_fraction_mean")),
                    _format_pct(_to_float(oracle, "correct_fraction_mean")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Winner-Only Best Fixed Results")
    lines.append("")
    lines.append("| Dataset | Arch | Attack | Winner | Pos Cert | Correct | Max Radius |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for combo in sorted(winner_fixed_best):
        row = winner_fixed_best[combo]
        winner = f"{row['model_variant']} / {row['config_label']} / thr={row['threshold']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["model_architecture"],
                    row["attack_mode"],
                    winner,
                    _format_pct(_to_float(row, "positive_certified_fraction_mean")),
                    _format_pct(_to_float(row, "correct_fraction_mean")),
                    _format_num(_to_float(row, "max_reported_radius_mean")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Selector Vs Oracle")
    lines.append("")
    lines.append("| Dataset | Arch | Attack | Selector Thr | Selector Pos Cert | Oracle Pos Cert | Gap |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for combo in sorted(winner_selector_best):
        selector = winner_selector_best[combo]
        oracle = winner_oracle_best[combo]
        gap = _to_float(oracle, "positive_certified_fraction_mean") - _to_float(selector, "positive_certified_fraction_mean")
        lines.append(
            "| "
            + " | ".join(
                [
                    selector["dataset"],
                    selector["model_architecture"],
                    selector["attack_mode"],
                    selector["threshold"],
                    _format_pct(_to_float(selector, "positive_certified_fraction_mean")),
                    _format_pct(_to_float(oracle, "positive_certified_fraction_mean")),
                    _format_pct(gap),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Target Coverage")
    lines.append("")
    lines.append("| Dataset | Arch | Attack | Variant | Evaluated Targets | Goal |")
    lines.append("|---|---|---|---|---:|---:|")
    for row in sorted(
        winner_pool_rows,
        key=lambda item: (
            item["dataset"],
            item["model_architecture"],
            item["attack_mode"],
            item["model_variant"],
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["model_architecture"],
                    row["attack_mode"],
                    row["model_variant"],
                    _format_num(_to_float(row, "evaluated_targets_mean")),
                    str(_to_int(row, "target_count_goal")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for figure_path in figure_paths:
        status = "present" if figure_path.exists() else "missing"
        lines.append(f"- {figure_path}: {status}")
    lines.append("")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Saved final report to {args.output_path.resolve()}")


if __name__ == "__main__":
    main()