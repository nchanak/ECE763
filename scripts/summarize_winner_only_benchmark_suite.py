from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


SUMMARY_FILENAMES = {
    "fixed": "winner_only_fixed_config_summary.csv",
    "selector": "winner_only_selector_summary.csv",
    "oracle": "winner_only_oracle_summary.csv",
    "target_pool": "winner_only_target_pool_summary.csv",
}


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


def _pick_best_fixed_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(
        rows,
        key=lambda row: (
            _to_float(row, "positive_certified_fraction_mean"),
            _to_float(row, "correct_fraction_mean"),
            _to_float(row, "mean_reported_radius_mean"),
            _to_float(row, "max_reported_radius_mean"),
            -_to_float(row, "abstained_fraction_mean"),
        ),
    )


def _pick_best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(
        rows,
        key=lambda row: (
            _to_float(row, "positive_certified_fraction_mean"),
            _to_float(row, "correct_fraction_mean"),
            _to_float(row, "mean_reported_radius_mean"),
            _to_float(row, "max_reported_radius_mean"),
            -_to_float(row, "abstained_fraction_mean"),
        ),
    )


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _threshold_label(row: dict[str, str]) -> str:
    return f"thr={row['threshold']}"


def _load_inputs(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    missing = [name for name in SUMMARY_FILENAMES.values() if not (output_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing required summary files in {output_dir}: {missing_list}")

    return {
        key: _read_csv_rows(output_dir / filename)
        for key, filename in SUMMARY_FILENAMES.items()
    }


def _build_fixed_winner_rows(fixed_rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Fixed-Config Winners",
        "",
        "| Dataset | Arch | Attack | Winner | Positive Cert | Correct | Mean Radius | Max Radius | Abstain |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    grouped = _group_rows(fixed_rows, ("dataset", "model_architecture", "attack_mode"))
    for key in sorted(grouped):
        best = _pick_best_fixed_row(grouped[key])
        winner = (
            f"{best['model_variant']} / {best['config_label']} / "
            f"{best['purification_operator']} / {best['certificate_report_strategy']} / {_threshold_label(best)}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    best["dataset"],
                    best["model_architecture"],
                    best["attack_mode"],
                    winner,
                    _format_pct(_to_float(best, "positive_certified_fraction_mean")),
                    _format_pct(_to_float(best, "correct_fraction_mean")),
                    _format_float(_to_float(best, "mean_reported_radius_mean")),
                    _format_float(_to_float(best, "max_reported_radius_mean")),
                    _format_pct(_to_float(best, "abstained_fraction_mean")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _build_selector_rows(
    selector_rows: list[dict[str, str]], oracle_rows: list[dict[str, str]]
) -> list[str]:
    lines = [
        "## Selector Vs Oracle",
        "",
        "| Dataset | Arch | Attack | Selector Threshold | Selector Positive Cert | Oracle Positive Cert | Gap | Selector Correct | Oracle Correct |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    selector_grouped = _group_rows(selector_rows, ("dataset", "model_architecture", "attack_mode"))
    oracle_grouped = _group_rows(oracle_rows, ("dataset", "model_architecture", "attack_mode"))
    for key in sorted(selector_grouped):
        selector_best = _pick_best_row(selector_grouped[key])
        oracle_best = _pick_best_row(oracle_grouped[key])
        gap = _to_float(oracle_best, "positive_certified_fraction_mean") - _to_float(
            selector_best, "positive_certified_fraction_mean"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    selector_best["dataset"],
                    selector_best["model_architecture"],
                    selector_best["attack_mode"],
                    selector_best["threshold"],
                    _format_pct(_to_float(selector_best, "positive_certified_fraction_mean")),
                    _format_pct(_to_float(oracle_best, "positive_certified_fraction_mean")),
                    _format_pct(gap),
                    _format_pct(_to_float(selector_best, "correct_fraction_mean")),
                    _format_pct(_to_float(oracle_best, "correct_fraction_mean")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _build_adaptive_delta_rows(fixed_rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Standard Vs Adaptive",
        "",
        "| Dataset | Arch | Best Standard Winner | Positive Cert | Best Adaptive Winner | Positive Cert | Delta |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    grouped = _group_rows(fixed_rows, ("dataset", "model_architecture", "attack_mode"))
    dataset_arch_keys = sorted({(row["dataset"], row["model_architecture"]) for row in fixed_rows})
    for dataset, architecture in dataset_arch_keys:
        standard_rows = grouped.get((dataset, architecture, "standard"))
        adaptive_rows = grouped.get((dataset, architecture, "adaptive-purified"))
        if not standard_rows or not adaptive_rows:
            continue
        standard_best = _pick_best_fixed_row(standard_rows)
        adaptive_best = _pick_best_fixed_row(adaptive_rows)
        standard_name = (
            f"{standard_best['model_variant']} / {standard_best['config_label']} / {_threshold_label(standard_best)}"
        )
        adaptive_name = (
            f"{adaptive_best['model_variant']} / {adaptive_best['config_label']} / {_threshold_label(adaptive_best)}"
        )
        delta = _to_float(adaptive_best, "positive_certified_fraction_mean") - _to_float(
            standard_best, "positive_certified_fraction_mean"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    architecture,
                    standard_name,
                    _format_pct(_to_float(standard_best, "positive_certified_fraction_mean")),
                    adaptive_name,
                    _format_pct(_to_float(adaptive_best, "positive_certified_fraction_mean")),
                    _format_pct(delta),
                ]
            )
            + " |"
        )
    if len(lines) == 4:
        lines.append("| No paired standard/adaptive results available yet | - | - | - | - | - | - |")
    lines.append("")
    return lines


def _build_target_pool_rows(target_pool_rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Attack Pool Coverage",
        "",
        "| Dataset | Arch | Attack | Variant | Pool Mode | Successful Attacks | Evaluated Targets | Goal |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(
        target_pool_rows,
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
                    row["target_pool_mode"],
                    _format_float(_to_float(row, "successful_attacks_mean")),
                    _format_float(_to_float(row, "evaluated_targets_mean")),
                    str(_to_int(row, "target_count_goal")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _build_failure_modes(
    fixed_rows: list[dict[str, str]],
    selector_rows: list[dict[str, str]],
    oracle_rows: list[dict[str, str]],
    target_pool_rows: list[dict[str, str]],
) -> list[str]:
    lines = ["## Failure Modes", ""]

    collapse_rows = [
        row
        for row in fixed_rows
        if _to_float(row, "correct_fraction_mean") > 0.0
        and _to_float(row, "positive_certified_fraction_mean") == 0.0
    ]
    if collapse_rows:
        lines.append(
            f"- Certificate collapse persists in {len(collapse_rows)} fixed-config settings: purified accuracy remains non-zero but the reported certified radius stays at 0."
        )

    fallback_rows = [row for row in target_pool_rows if row["target_pool_mode"] != "successful-attacks"]
    if fallback_rows:
        lines.append(
            f"- Attack-pool shortfall appears in {len(fallback_rows)} model/attack combinations where the run fell back from successful attacks to attempted targets."
        )

    low_coverage_rows = [
        row
        for row in target_pool_rows
        if _to_float(row, "evaluated_targets_mean") < _to_float(row, "target_count_goal")
    ]
    if low_coverage_rows:
        lines.append(
            f"- Target coverage is incomplete in {len(low_coverage_rows)} combinations: evaluated targets came in below the requested goal."
        )

    selector_grouped = _group_rows(selector_rows, ("dataset", "model_architecture", "attack_mode"))
    oracle_grouped = _group_rows(oracle_rows, ("dataset", "model_architecture", "attack_mode"))
    gap_rows: list[tuple[tuple[str, ...], float]] = []
    for key, rows in selector_grouped.items():
        selector_best = _pick_best_row(rows)
        oracle_best = _pick_best_row(oracle_grouped[key])
        gap = _to_float(oracle_best, "positive_certified_fraction_mean") - _to_float(
            selector_best, "positive_certified_fraction_mean"
        )
        if gap > 0.05:
            gap_rows.append((key, gap))
    if gap_rows:
        worst_key, worst_gap = max(gap_rows, key=lambda item: item[1])
        lines.append(
            "- Selector quality is still leaving value on the table in "
            f"{len(gap_rows)} combinations; the worst oracle gap is {worst_gap * 100:.1f}% on "
            f"{worst_key[0]} / {worst_key[1]} / {worst_key[2]}."
        )

    if len(lines) == 2:
        lines.append("- No dominant failure mode crossed the configured thresholds in the aggregate summaries.")

    lines.append("")
    return lines


def _build_key_findings(
    fixed_rows: list[dict[str, str]],
    selector_rows: list[dict[str, str]],
    oracle_rows: list[dict[str, str]],
) -> list[str]:
    lines = ["## Key Findings", ""]

    overall_best_fixed = _pick_best_fixed_row(fixed_rows)
    lines.append(
        "- Best fixed configuration overall: "
        f"{overall_best_fixed['dataset']} / {overall_best_fixed['model_architecture']} / {overall_best_fixed['attack_mode']} / "
        f"{overall_best_fixed['model_variant']} / {overall_best_fixed['config_label']} / {_threshold_label(overall_best_fixed)} "
        f"with positive certification {_format_pct(_to_float(overall_best_fixed, 'positive_certified_fraction_mean'))}, "
        f"correctness {_format_pct(_to_float(overall_best_fixed, 'correct_fraction_mean'))}, and mean radius {_format_float(_to_float(overall_best_fixed, 'mean_reported_radius_mean'))}."
    )

    overall_best_selector = _pick_best_row(selector_rows)
    overall_best_oracle = _pick_best_row(oracle_rows)
    lines.append(
        "- Best selector summary overall: "
        f"{overall_best_selector['dataset']} / {overall_best_selector['model_architecture']} / {overall_best_selector['attack_mode']} / "
        f"thr={overall_best_selector['threshold']} at positive certification {_format_pct(_to_float(overall_best_selector, 'positive_certified_fraction_mean'))}."
    )
    lines.append(
        "- Oracle headroom overall: "
        f"best oracle reaches {_format_pct(_to_float(overall_best_oracle, 'positive_certified_fraction_mean'))}, "
        f"a {_format_pct(_to_float(overall_best_oracle, 'positive_certified_fraction_mean') - _to_float(overall_best_selector, 'positive_certified_fraction_mean'))} lift over the best selector point."
    )
    lines.append("")
    return lines


def build_markdown(output_dir: Path) -> str:
    inputs = _load_inputs(output_dir)
    fixed_rows = inputs["fixed"]
    selector_rows = inputs["selector"]
    oracle_rows = inputs["oracle"]
    target_pool_rows = inputs["target_pool"]

    lines = [
        "# Winner-Only Benchmark Suite Summary",
        "",
        f"Output directory: {output_dir}",
        "",
        _build_key_findings(fixed_rows, selector_rows, oracle_rows),
        _build_fixed_winner_rows(fixed_rows),
        _build_selector_rows(selector_rows, oracle_rows),
        _build_adaptive_delta_rows(fixed_rows),
        _build_target_pool_rows(target_pool_rows),
        _build_failure_modes(fixed_rows, selector_rows, oracle_rows, target_pool_rows),
    ]
    flat_lines: list[str] = []
    for block in lines:
        if isinstance(block, list):
            flat_lines.extend(block)
        else:
            flat_lines.append(block)
    return "\n".join(flat_lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize winner-only benchmark suite outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "winner_only_benchmark_suite_v1",
        help="Directory containing winner_only_*_summary.csv files.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Path to write the markdown summary. Defaults to <output-dir>/winner_only_benchmark_suite_summary.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else output_dir / "winner_only_benchmark_suite_summary.md"
    markdown = build_markdown(output_dir)
    summary_path.write_text(markdown, encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()