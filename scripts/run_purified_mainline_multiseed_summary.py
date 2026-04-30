import argparse
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
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main_entry():
    parser = argparse.ArgumentParser(description="Run the revised purified mainline multiseed summary")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        default="results/purified_mainline_multiseed_v2",
        help="Directory for multiseed purified certificate CSV outputs",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dataset = Planetoid(root="data/Planetoid", name=args.dataset, transform=NormalizeFeatures())
    data = dataset[0].to(device)

    bundle = main._run_multiseed_purified_certificate_summary(dataset, data, device)
    output_dir = main.ensure_results_dir(args.output_dir)

    main.save_csv_rows(output_dir / "purified_certificate_multiseed_target_pools.csv", bundle["target_pool_rows"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_target_pool_summary.csv", bundle["target_pool_summary"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_fixed_config_per_seed.csv", bundle["fixed_config_per_seed_rows"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_fixed_config_summary.csv", bundle["fixed_config_summary"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_ablation_per_seed.csv", bundle["ablation_per_seed_rows"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_ablation_summary.csv", bundle["ablation_summary"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_selector_per_seed.csv", bundle["selector_per_seed_rows"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_selector_summary.csv", bundle["selector_summary"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_oracle_per_seed.csv", bundle["oracle_per_seed_rows"])
    main.save_csv_rows(output_dir / "purified_certificate_multiseed_oracle_summary.csv", bundle["oracle_summary"])

    print({"saved_to": str(Path(output_dir).resolve()), "device": str(device)})


if __name__ == "__main__":
    main_entry()