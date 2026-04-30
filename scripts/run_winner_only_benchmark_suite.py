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

                print(
                    f"Running dataset={dataset_name} arch={model_architecture} attack={attack_mode} "
                    f"seeds={args.seeds} targets={args.purified_target_count}"
                )

                bundle = main._run_multiseed_purified_certificate_summary(
                    dataset=dataset,
                    data=data,
                    device=device,
                    model_architecture=model_architecture,
                    seeds=args.seeds,
                    target_count=args.purified_target_count,
                    candidate_sweep=main.WINNER_ONLY_PURIFIED_CERTIFICATE_CANDIDATE_SWEEP,
                    ablation_sweep=[],
                    variant_labels=main.WINNER_ONLY_TRAINING_VARIANT_LABELS,
                    attack_mode=attack_mode,
                    adaptive_purification_thresholds=args.adaptive_thresholds,
                )

                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_target_pools.csv", bundle["target_pool_rows"])
                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_target_pool_summary.csv", bundle["target_pool_summary"])
                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_fixed_config_per_seed.csv", bundle["fixed_config_per_seed_rows"])
                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_fixed_config_summary.csv", bundle["fixed_config_summary"])
                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_selector_per_seed.csv", bundle["selector_per_seed_rows"])
                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_selector_summary.csv", bundle["selector_summary"])
                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_oracle_per_seed.csv", bundle["oracle_per_seed_rows"])
                main.save_csv_rows(combo_output_dir / "purified_certificate_multiseed_oracle_summary.csv", bundle["oracle_summary"])

                combined_target_pool_summary.extend(
                    _augment_rows(bundle["target_pool_summary"], dataset_name, model_architecture, attack_mode)
                )
                combined_fixed_config_summary.extend(
                    _augment_rows(bundle["fixed_config_summary"], dataset_name, model_architecture, attack_mode)
                )
                combined_selector_summary.extend(
                    _augment_rows(bundle["selector_summary"], dataset_name, model_architecture, attack_mode)
                )
                combined_oracle_summary.extend(
                    _augment_rows(bundle["oracle_summary"], dataset_name, model_architecture, attack_mode)
                )

                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()

    main.save_csv_rows(output_root / "winner_only_target_pool_summary.csv", combined_target_pool_summary)
    main.save_csv_rows(output_root / "winner_only_fixed_config_summary.csv", combined_fixed_config_summary)
    main.save_csv_rows(output_root / "winner_only_selector_summary.csv", combined_selector_summary)
    main.save_csv_rows(output_root / "winner_only_oracle_summary.csv", combined_oracle_summary)

    print({"saved_to": str(output_root.resolve()), "device": str(device)})


if __name__ == "__main__":
    main_entry()