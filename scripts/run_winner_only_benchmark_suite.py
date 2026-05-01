import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import main


def _resolve_device(device_name: str) -> torch.device:
    normalized_device = str(device_name or "auto").strip().lower()
    if normalized_device == "cpu":
        return torch.device("cpu")
    if normalized_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_csv_list(raw_value: str):
    values = [item.strip() for item in str(raw_value or "").split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list with at least one value")
    return values


def _normalize_dataset_name(dataset_name: str) -> str:
    aliases = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
    }
    normalized = str(dataset_name or "").strip().lower()
    if normalized not in aliases:
        raise ValueError(f"Unsupported Planetoid dataset: {dataset_name}")
    return aliases[normalized]


def _augment_rows(rows, dataset_name: str, model_architecture: str, attack_mode: str):
    return [
        {
            "dataset": dataset_name,
            "model_architecture": model_architecture,
            "attack_mode": attack_mode,
            **row,
        }
        for row in rows
    ]


COMBO_SUMMARY_FILE_NAMES = {
    "target_pool_summary": "purified_certificate_multiseed_target_pool_summary.csv",
    "fixed_config_summary": "purified_certificate_multiseed_fixed_config_summary.csv",
    "selector_summary": "purified_certificate_multiseed_selector_summary.csv",
    "oracle_summary": "purified_certificate_multiseed_oracle_summary.csv",
}

COMBO_PER_SEED_FILE_NAMES = {
    "target_pool_rows": "purified_certificate_multiseed_target_pools.csv",
    "fixed_config_per_seed": "purified_certificate_multiseed_fixed_config_per_seed.csv",
    "selector_per_seed": "purified_certificate_multiseed_selector_per_seed.csv",
    "oracle_per_seed": "purified_certificate_multiseed_oracle_per_seed.csv",
}

COMBO_REQUIRED_FILE_NAMES = {
    **COMBO_SUMMARY_FILE_NAMES,
    **COMBO_PER_SEED_FILE_NAMES,
}


def _load_csv_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stringify_rows(rows):
    return [
        {
            key: ("" if value is None else str(value))
            for key, value in row.items()
        }
        for row in rows
    ]


def _read_seed_values(csv_path: Path):
    rows = _load_csv_rows(csv_path)
    seed_values = set()
    for row in rows:
        raw_seed = row.get("seed")
        if raw_seed is None or raw_seed == "":
            continue
        seed_values.add(int(raw_seed))
    return seed_values


def _has_completed_combo_outputs(combo_output_dir: Path, expected_seeds=None) -> bool:
    if not all((combo_output_dir / file_name).exists() for file_name in COMBO_REQUIRED_FILE_NAMES.values()):
        return False

    if expected_seeds is None:
        return True

    requested_seed_values = {int(seed) for seed in expected_seeds}
    completed_seed_values = _read_seed_values(combo_output_dir / COMBO_PER_SEED_FILE_NAMES["fixed_config_per_seed"])
    return completed_seed_values == requested_seed_values


def _load_completed_combo_summaries(combo_output_dir: Path):
    if not _has_completed_combo_outputs(combo_output_dir):
        raise FileNotFoundError(f"Missing completed combo summary outputs under {combo_output_dir}")
    return {
        key: _load_csv_rows(combo_output_dir / file_name)
        for key, file_name in COMBO_SUMMARY_FILE_NAMES.items()
    }


def _load_all_completed_suite_summaries(output_root: Path, expected_seeds=None):
    combined_target_pool_summary = []
    combined_fixed_config_summary = []
    combined_selector_summary = []
    combined_oracle_summary = []

    if not output_root.exists():
        return {
            "target_pool_summary": combined_target_pool_summary,
            "fixed_config_summary": combined_fixed_config_summary,
            "selector_summary": combined_selector_summary,
            "oracle_summary": combined_oracle_summary,
        }

    for dataset_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        for architecture_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            model_architecture = architecture_dir.name
            for attack_dir in sorted(path for path in architecture_dir.iterdir() if path.is_dir()):
                combo_output_dir = attack_dir
                if not _has_completed_combo_outputs(combo_output_dir, expected_seeds=expected_seeds):
                    continue

                attack_mode = attack_dir.name.replace("_", "-")
                completed_summaries = _load_completed_combo_summaries(combo_output_dir)
                combined_target_pool_summary.extend(
                    _augment_rows(completed_summaries["target_pool_summary"], dataset_name, model_architecture, attack_mode)
                )
                combined_fixed_config_summary.extend(
                    _augment_rows(completed_summaries["fixed_config_summary"], dataset_name, model_architecture, attack_mode)
                )
                combined_selector_summary.extend(
                    _augment_rows(completed_summaries["selector_summary"], dataset_name, model_architecture, attack_mode)
                )
                combined_oracle_summary.extend(
                    _augment_rows(completed_summaries["oracle_summary"], dataset_name, model_architecture, attack_mode)
                )

    return {
        "target_pool_summary": combined_target_pool_summary,
        "fixed_config_summary": combined_fixed_config_summary,
        "selector_summary": combined_selector_summary,
        "oracle_summary": combined_oracle_summary,
    }


def _save_root_suite_summaries(output_root: Path, expected_seeds):
    completed_suite_summaries = _load_all_completed_suite_summaries(output_root, expected_seeds=expected_seeds)
    main.save_csv_rows(output_root / "winner_only_target_pool_summary.csv", completed_suite_summaries["target_pool_summary"])
    main.save_csv_rows(output_root / "winner_only_fixed_config_summary.csv", completed_suite_summaries["fixed_config_summary"])
    main.save_csv_rows(output_root / "winner_only_selector_summary.csv", completed_suite_summaries["selector_summary"])
    main.save_csv_rows(output_root / "winner_only_oracle_summary.csv", completed_suite_summaries["oracle_summary"])


TARGET_POOL_GROUP_KEYS = ["model_variant", "attack_variant", "target_pool_mode", "target_count_goal"]
TARGET_POOL_METRIC_KEYS = ["attempted_nodes", "successful_attacks", "evaluated_targets"]

FIXED_CONFIG_GROUP_KEYS = [
    "config_branch",
    "model_variant",
    "attack_variant",
    "threshold",
    "config_label",
    "mode",
    "purification_operator",
    "purification_threshold",
    "certificate_report_strategy",
    "adaptive_profile",
    "p_delete",
    "p_add",
    "max_additions",
]

FIXED_CONFIG_METRIC_KEYS = [
    "evaluated_nodes",
    "purified_plain_correct_fraction",
    "correct_fraction",
    "abstained_fraction",
    "correct_abstained_fraction",
    "correct_zero_radius_fraction",
    "positive_certified_fraction",
    "max_reported_radius",
    "mean_pA_lower",
    "mean_lower_margin",
    "mean_purified_confidence",
    "mean_purified_margin",
    "mean_target_degree",
    "mean_resolved_p_delete",
    "mean_resolved_p_add",
    "mean_resolved_max_additions",
]

SELECTOR_ORACLE_GROUP_KEYS = ["config_branch", "model_variant", "attack_variant", "threshold"]

SELECTOR_ORACLE_METRIC_KEYS = [
    "evaluated_nodes",
    "purified_plain_correct_fraction",
    "correct_fraction",
    "abstained_fraction",
    "correct_abstained_fraction",
    "correct_zero_radius_fraction",
    "positive_certified_fraction",
    "mean_reported_radius",
    "max_reported_radius",
    "mean_pA_lower",
    "mean_lower_margin",
    "mean_target_edge_retention",
    "mean_purified_confidence",
    "mean_purified_margin",
    "mean_target_degree",
    "nodes_with_correct_config",
    "nodes_without_correct_config",
]


def _aggregate_multiseed_rows(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in grouped.items():
        summary_row = {key_name: key_value for key_name, key_value in zip(group_keys, key)}
        num_seeds = int(len(group_rows))
        summary_row["num_seeds"] = num_seeds
        summary_row["seeds"] = "; ".join(
            str(row["seed"]) for row in sorted(group_rows, key=lambda row: int(row["seed"]))
        )
        for metric_key in metric_keys:
            values = main.np.array([float(row[metric_key]) for row in group_rows], dtype=float)
            mean_value = float(values.mean())
            population_std = float(values.std())
            sample_std = float(values.std(ddof=1)) if num_seeds > 1 else 0.0
            sem = sample_std / main.np.sqrt(num_seeds) if num_seeds > 1 else 0.0
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


def _load_combo_existing_rows(combo_output_dir: Path):
    rows_by_key = {}
    for key, file_name in COMBO_PER_SEED_FILE_NAMES.items():
        csv_path = combo_output_dir / file_name
        rows_by_key[key] = _load_csv_rows(csv_path) if csv_path.exists() else []
    return rows_by_key


def _replace_seed_rows(existing_rows, new_rows, seed):
    seed_value = int(seed)
    filtered_rows = [row for row in existing_rows if int(row["seed"]) != seed_value]
    filtered_rows.extend(_stringify_rows(new_rows))
    return filtered_rows


def _sort_per_seed_rows(rows, *, extra_keys):
    def _sort_key(row):
        return (int(row["seed"]),) + tuple(str(row.get(key, "")) for key in extra_keys)

    return sorted(rows, key=_sort_key)


def _save_combo_outputs(combo_output_dir: Path, rows_by_key):
    target_pool_rows = _sort_per_seed_rows(
        rows_by_key["target_pool_rows"],
        extra_keys=["model_variant", "attack_variant", "target_pool_mode"],
    )
    fixed_config_per_seed_rows = _sort_per_seed_rows(
        rows_by_key["fixed_config_per_seed"],
        extra_keys=["model_variant", "attack_variant", "threshold", "config_label", "config_branch"],
    )
    selector_per_seed_rows = _sort_per_seed_rows(
        rows_by_key["selector_per_seed"],
        extra_keys=["model_variant", "attack_variant", "threshold", "summary_type"],
    )
    oracle_per_seed_rows = _sort_per_seed_rows(
        rows_by_key["oracle_per_seed"],
        extra_keys=["model_variant", "attack_variant", "threshold", "summary_type"],
    )

    target_pool_summary = _aggregate_multiseed_rows(target_pool_rows, TARGET_POOL_GROUP_KEYS, TARGET_POOL_METRIC_KEYS)
    fixed_config_summary = _aggregate_multiseed_rows(
        fixed_config_per_seed_rows,
        FIXED_CONFIG_GROUP_KEYS,
        FIXED_CONFIG_METRIC_KEYS,
    )
    selector_summary = _aggregate_multiseed_rows(
        selector_per_seed_rows,
        SELECTOR_ORACLE_GROUP_KEYS,
        SELECTOR_ORACLE_METRIC_KEYS,
    )
    oracle_summary = _aggregate_multiseed_rows(
        oracle_per_seed_rows,
        SELECTOR_ORACLE_GROUP_KEYS,
        SELECTOR_ORACLE_METRIC_KEYS,
    )

    main.save_csv_rows(combo_output_dir / COMBO_PER_SEED_FILE_NAMES["target_pool_rows"], target_pool_rows)
    main.save_csv_rows(combo_output_dir / COMBO_SUMMARY_FILE_NAMES["target_pool_summary"], target_pool_summary)
    main.save_csv_rows(combo_output_dir / COMBO_PER_SEED_FILE_NAMES["fixed_config_per_seed"], fixed_config_per_seed_rows)
    main.save_csv_rows(combo_output_dir / COMBO_SUMMARY_FILE_NAMES["fixed_config_summary"], fixed_config_summary)
    main.save_csv_rows(combo_output_dir / COMBO_PER_SEED_FILE_NAMES["selector_per_seed"], selector_per_seed_rows)
    main.save_csv_rows(combo_output_dir / COMBO_SUMMARY_FILE_NAMES["selector_summary"], selector_summary)
    main.save_csv_rows(combo_output_dir / COMBO_PER_SEED_FILE_NAMES["oracle_per_seed"], oracle_per_seed_rows)
    main.save_csv_rows(combo_output_dir / COMBO_SUMMARY_FILE_NAMES["oracle_summary"], oracle_summary)


def main_entry():
    parser = argparse.ArgumentParser(description="Run the reduced winner-only purified benchmark suite")
    parser.add_argument("--datasets", type=_parse_csv_list, default=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument(
        "--architectures",
        type=_parse_csv_list,
        default=[main.DEFAULT_MODEL_ARCHITECTURE, "graphsage"],
        help="Comma-separated model architectures, e.g. gcn,graphsage",
    )
    parser.add_argument(
        "--attack-modes",
        type=_parse_csv_list,
        default=["standard", "adaptive-purified"],
        help="Comma-separated attack modes, e.g. standard,adaptive-purified",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="results/winner_only_benchmark_suite_v1")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-combination summary CSVs instead of rerunning those combinations",
    )
    parser.add_argument(
        "--seeds",
        type=main._parse_seed_list,
        default=main.WINNER_ONLY_MULTI_SEED_SUMMARY_SEEDS,
        help="Comma-separated seeds for the reduced multiseed run",
    )
    parser.add_argument(
        "--purified-target-count",
        type=int,
        default=main.WINNER_ONLY_PURIFIED_TARGET_COUNT,
        help="Successful targeted attacks to evaluate per model/seed",
    )
    parser.add_argument(
        "--adaptive-thresholds",
        type=main._parse_float_list,
        default=main.ADAPTIVE_PURIFIED_ATTACK_THRESHOLDS,
        help="Comma-separated purification thresholds for the adaptive targeted attack",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    output_root = main.ensure_results_dir(args.output_dir)

    combined_target_pool_summary = []
    combined_fixed_config_summary = []
    combined_selector_summary = []
    combined_oracle_summary = []

    normalized_datasets = [_normalize_dataset_name(dataset_name) for dataset_name in args.datasets]
    normalized_architectures = [str(architecture).strip().lower() for architecture in args.architectures]
    normalized_attack_modes = [str(attack_mode).strip().lower() for attack_mode in args.attack_modes]

    for dataset_name in normalized_datasets:
        dataset = Planetoid(root="data/Planetoid", name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0].to(device)

        for model_architecture in normalized_architectures:
            if model_architecture not in main.MODEL_ARCHITECTURE_CHOICES:
                raise ValueError(f"Unsupported model architecture: {model_architecture}")

            for attack_mode in normalized_attack_modes:
                combo_output_dir = output_root / dataset_name / model_architecture / attack_mode.replace("-", "_")
                combo_output_dir.mkdir(parents=True, exist_ok=True)

                if args.resume and _has_completed_combo_outputs(combo_output_dir, expected_seeds=args.seeds):
                    print(
                        f"Skipping completed dataset={dataset_name} arch={model_architecture} attack={attack_mode} "
                        f"from {combo_output_dir}"
                    )
                    completed_summaries = _load_completed_combo_summaries(combo_output_dir)
                    combined_target_pool_summary.extend(
                        _augment_rows(completed_summaries["target_pool_summary"], dataset_name, model_architecture, attack_mode)
                    )
                    combined_fixed_config_summary.extend(
                        _augment_rows(completed_summaries["fixed_config_summary"], dataset_name, model_architecture, attack_mode)
                    )
                    combined_selector_summary.extend(
                        _augment_rows(completed_summaries["selector_summary"], dataset_name, model_architecture, attack_mode)
                    )
                    combined_oracle_summary.extend(
                        _augment_rows(completed_summaries["oracle_summary"], dataset_name, model_architecture, attack_mode)
                    )
                    continue

                if args.resume and combo_output_dir.exists() and any(combo_output_dir.iterdir()):
                    print(
                        f"Recomputing incomplete dataset={dataset_name} arch={model_architecture} attack={attack_mode} "
                        f"from {combo_output_dir}"
                    )

                existing_rows = _load_combo_existing_rows(combo_output_dir) if args.resume else {
                    key: [] for key in COMBO_PER_SEED_FILE_NAMES
                }
                completed_seed_values = {
                    int(row["seed"])
                    for row in existing_rows["fixed_config_per_seed"]
                    if row.get("seed") not in {None, ""}
                }
                pending_seeds = [int(seed) for seed in args.seeds if int(seed) not in completed_seed_values]

                if pending_seeds:
                    print(
                        f"Running dataset={dataset_name} arch={model_architecture} attack={attack_mode} "
                        f"pending_seeds={pending_seeds} targets={args.purified_target_count}"
                    )

                for seed in pending_seeds:
                    print(
                        f"Starting dataset={dataset_name} arch={model_architecture} attack={attack_mode} seed={seed}"
                    )
                    bundle = main._run_multiseed_purified_certificate_summary(
                        dataset=dataset,
                        data=data,
                        device=device,
                        model_architecture=model_architecture,
                        seeds=[seed],
                        target_count=args.purified_target_count,
                        candidate_sweep=main.WINNER_ONLY_PURIFIED_CERTIFICATE_CANDIDATE_SWEEP,
                        ablation_sweep=[],
                        variant_labels=main.WINNER_ONLY_TRAINING_VARIANT_LABELS,
                        attack_mode=attack_mode,
                        adaptive_purification_thresholds=args.adaptive_thresholds,
                    )

                    existing_rows["target_pool_rows"] = _replace_seed_rows(
                        existing_rows["target_pool_rows"],
                        bundle["target_pool_rows"],
                        seed,
                    )
                    existing_rows["fixed_config_per_seed"] = _replace_seed_rows(
                        existing_rows["fixed_config_per_seed"],
                        bundle["fixed_config_per_seed_rows"],
                        seed,
                    )
                    existing_rows["selector_per_seed"] = _replace_seed_rows(
                        existing_rows["selector_per_seed"],
                        bundle["selector_per_seed_rows"],
                        seed,
                    )
                    existing_rows["oracle_per_seed"] = _replace_seed_rows(
                        existing_rows["oracle_per_seed"],
                        bundle["oracle_per_seed_rows"],
                        seed,
                    )
                    _save_combo_outputs(combo_output_dir, existing_rows)
                    _save_root_suite_summaries(output_root, expected_seeds=args.seeds)
                    completed_seed_values.add(int(seed))
                    print(
                        f"Checkpointed dataset={dataset_name} arch={model_architecture} attack={attack_mode} "
                        f"completed_seeds={sorted(completed_seed_values)}"
                    )

                    if torch.cuda.is_available() and device.type == "cuda":
                        torch.cuda.empty_cache()

                if _has_completed_combo_outputs(combo_output_dir, expected_seeds=args.seeds):
                    completed_summaries = _load_completed_combo_summaries(combo_output_dir)
                    combined_target_pool_summary.extend(
                        _augment_rows(completed_summaries["target_pool_summary"], dataset_name, model_architecture, attack_mode)
                    )
                    combined_fixed_config_summary.extend(
                        _augment_rows(completed_summaries["fixed_config_summary"], dataset_name, model_architecture, attack_mode)
                    )
                    combined_selector_summary.extend(
                        _augment_rows(completed_summaries["selector_summary"], dataset_name, model_architecture, attack_mode)
                    )
                    combined_oracle_summary.extend(
                        _augment_rows(completed_summaries["oracle_summary"], dataset_name, model_architecture, attack_mode)
                    )

    _save_root_suite_summaries(output_root, expected_seeds=args.seeds)

    print({"saved_to": str(output_root.resolve()), "device": str(device)})


if __name__ == "__main__":
    main_entry()