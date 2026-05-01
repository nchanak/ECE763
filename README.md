Install the dependencies in [requirements.txt](requirements.txt) before running the scripts.

## Current Project State

- [main.py](main.py) runs the full graph-robustness pipeline: clean and robustness-aligned training variants, PRBCD global attack, DeeProBUST Nettack targeted attack, Jaccard and cosine purification, smoothing and certification, and artifact generation.

- The project now supports both the original Cora-focused pipeline and the completed winner-only cross-dataset benchmark over Cora, CiteSeer, and PubMed using both GCN and GraphSAGE.

- Attacks live in [src/attack.py](src/attack.py) and [src/nettack.py](src/nettack.py).

- Smoothing and certification logic lives in [src/smoothing.py](src/smoothing.py), with training and evaluation helpers in [src/train.py](src/train.py).

- Target-node purification lives in [src/purification.py](src/purification.py).

- The authoritative final summary is [results/final_report.md](results/final_report.md).

- The completed winner-only suite lives under [results/winner_only_benchmark_suite_v1](results/winner_only_benchmark_suite_v1), with figures in [results/winner_only_benchmark_suite_v1/figures](results/winner_only_benchmark_suite_v1/figures).

- The old draft at [results/final_report_draft.md](results/final_report_draft.md) should not be treated as authoritative.

- The imported snapshot under [results/cert_debug_seed_42](results/cert_debug_seed_42) is useful for certificate-diagnostic figures, but it is not the authoritative result source for the final report tables.

## Key Result Artifacts

- Final narrative summary: [results/final_report.md](results/final_report.md)
- Training baselines: [results/training_variant_summary.csv](results/training_variant_summary.csv)
- Global attack curve: [results/attack_budget_accuracy.csv](results/attack_budget_accuracy.csv) and [results/attack_budget_accuracy.png](results/attack_budget_accuracy.png)
- Cora purification tradeoff: [results/purification_sweep.csv](results/purification_sweep.csv) and [results/purification_tradeoff.png](results/purification_tradeoff.png)
- Nettack target repair summary: [results/nettack_target_purification_sweep.csv](results/nettack_target_purification_sweep.csv) and [results/nettack_target_purification_rows.csv](results/nettack_target_purification_rows.csv)
- Winner-only fixed-config summary: [results/winner_only_benchmark_suite_v1/winner_only_fixed_config_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_fixed_config_summary.csv)
- Winner-only selector summary: [results/winner_only_benchmark_suite_v1/winner_only_selector_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_selector_summary.csv)
- Winner-only oracle summary: [results/winner_only_benchmark_suite_v1/winner_only_oracle_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_oracle_summary.csv)
- Winner-only target-pool coverage: [results/winner_only_benchmark_suite_v1/winner_only_target_pool_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_target_pool_summary.csv)
- Winner-only suite markdown summary: [results/winner_only_benchmark_suite_v1/winner_only_benchmark_suite_summary.md](results/winner_only_benchmark_suite_v1/winner_only_benchmark_suite_summary.md)

## CLI And Configurable Runs

- [main.py](main.py) accepts CLI configuration for dataset, model architecture, seeds, output directory, training epochs, attack budgets, edge-drop sweep, sparse flip sweep, certificate node count, certificate radius, asymmetric delete and add limits, Nettack target count, and smoothing-mode selection.

- Single-seed example:

```bash
python main.py --seed 42 --output-dir results/final_seed_42
```

- Multi-seed example:

```bash
python main.py --seeds 42,84,99 --output-dir results/multiseed
```

- Sparse sweep override example:

```bash
python main.py --sparse-flip-sweep "0.01:0.000005:64;0.02:0.000010:96;0.05:0.000020:160"
```

- The final winner-only suite and summary tooling are driven by [scripts/run_winner_only_benchmark_suite.py](scripts/run_winner_only_benchmark_suite.py), [scripts/summarize_winner_only_benchmark_suite.py](scripts/summarize_winner_only_benchmark_suite.py), and [scripts/compile_final_report.py](scripts/compile_final_report.py).

## Final Validated Result Snapshot

### Status

- Winner-only completed combinations: 12 / 12
- Winner-only datasets present: CiteSeer, Cora, PubMed
- Winner-only cross-dataset coverage is complete

### Training Baselines

| Variant | Clean Test | Best Val | Global Attack @ 50 |
|---|---:|---:|---:|
| clean-training | 0.817 | 0.796 | 0.791 |
| matched-sparse-noisy-training | 0.817 | 0.806 | 0.770 |
| clean-warmstart-symmetric-finetune | 0.809 | 0.798 | 0.777 |

### Cora Mainline Best Results

| Variant | Best Fixed | Fixed Thr | Fixed Pos Cert | Fixed Correct | Best Oracle Thr | Oracle Pos Cert | Oracle Correct |
|---|---|---:|---:|---:|---:|---:|---:|
| balanced-sparse-noisy-training | symmetric-0.3 | 0.01 | 13.9% | 13.9% | 0.01 | 13.9% | 13.9% |
| clean-training | symmetric-0.3-top2 | 0.05 | 29.5% | 43.4% | 0.05 | 29.5% | 46.2% |
| clean-warmstart-symmetric-finetune | symmetric-0.3-top2 | 0.02 | 29.5% | 43.4% | 0.01 | 29.5% | 46.2% |
| matched-sparse-noisy-training | symmetric-0.3-top2 | 0.05 | 19.4% | 33.3% | 0.05 | 19.4% | 41.7% |
| purification-aware-symmetric-finetune | symmetric-0.3-cosine-top2 | 0.02 | 29.5% | 43.4% | 0.05 | 29.5% | 46.2% |

### Winner-Only Best Fixed Results

| Dataset | Arch | Attack | Winner | Pos Cert | Correct | Max Radius |
|---|---|---|---|---:|---:|---:|
| CiteSeer | gcn | adaptive-purified | clean-training / symmetric-0.3 / thr=0.01 | 17.9% | 20.2% | 4.000 |
| CiteSeer | gcn | standard | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / thr=0.02 | 18.0% | 22.0% | 4.600 |
| CiteSeer | graphsage | adaptive-purified | clean-training / symmetric-0.3 / thr=0.01 | 43.2% | 46.0% | 5.000 |
| CiteSeer | graphsage | standard | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / thr=0.05 | 56.9% | 59.4% | 5.000 |
| Cora | gcn | adaptive-purified | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / thr=0.05 | 22.4% | 29.6% | 2.400 |
| Cora | gcn | standard | clean-training / symmetric-0.3-cosine-top2 / thr=0.01 | 24.4% | 32.5% | 4.000 |
| Cora | graphsage | adaptive-purified | clean-training / symmetric-0.3 / thr=0.01 | 48.5% | 48.5% | 4.000 |
| Cora | graphsage | standard | clean-training / symmetric-0.3 / thr=0.02 | 59.3% | 61.8% | 5.000 |
| PubMed | gcn | adaptive-purified | clean-training / symmetric-0.3 / thr=0.01 | 48.2% | 48.2% | 5.000 |
| PubMed | gcn | standard | clean-training / symmetric-0.3 / thr=0.01 | 48.0% | 48.0% | 5.000 |
| PubMed | graphsage | adaptive-purified | clean-training / symmetric-0.3-top2 / thr=0.01 | 73.8% | 77.2% | 5.000 |
| PubMed | graphsage | standard | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / thr=0.05 | 72.9% | 76.4% | 5.000 |

### Selector Vs Oracle

| Dataset | Arch | Attack | Selector Thr | Selector Pos Cert | Oracle Pos Cert | Gap |
|---|---|---|---:|---:|---:|---:|
| CiteSeer | gcn | adaptive-purified | 0.05 | 17.9% | 17.9% | 0.0% |
| CiteSeer | gcn | standard | 0.05 | 18.0% | 18.0% | 0.0% |
| CiteSeer | graphsage | adaptive-purified | 0.01 | 43.2% | 43.2% | 0.0% |
| CiteSeer | graphsage | standard | 0.02 | 55.7% | 56.9% | 1.2% |
| Cora | gcn | adaptive-purified | 0.05 | 22.4% | 22.4% | 0.0% |
| Cora | gcn | standard | 0.01 | 24.4% | 24.4% | 0.0% |
| Cora | graphsage | adaptive-purified | 0.01 | 48.5% | 48.5% | 0.0% |
| Cora | graphsage | standard | 0.02 | 59.3% | 59.3% | 0.0% |
| PubMed | gcn | adaptive-purified | 0.01 | 48.2% | 48.2% | 0.0% |
| PubMed | gcn | standard | 0.01 | 48.0% | 48.0% | 0.0% |
| PubMed | graphsage | adaptive-purified | 0.02 | 73.8% | 73.8% | 0.0% |
| PubMed | graphsage | standard | 0.01 | 72.9% | 72.9% | 0.0% |

### Standard Vs Adaptive

| Dataset | Arch | Best Standard Winner | Standard Pos Cert | Best Adaptive Winner | Adaptive Pos Cert | Delta |
|---|---|---|---:|---|---:|---:|
| CiteSeer | gcn | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / thr=0.02 | 18.0% | clean-training / symmetric-0.3 / thr=0.01 | 17.9% | -0.1% |
| CiteSeer | graphsage | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / thr=0.05 | 56.9% | clean-training / symmetric-0.3 / thr=0.01 | 43.2% | -13.7% |
| Cora | gcn | clean-training / symmetric-0.3-cosine-top2 / thr=0.01 | 24.4% | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / thr=0.05 | 22.4% | -2.1% |
| Cora | graphsage | clean-training / symmetric-0.3 / thr=0.02 | 59.3% | clean-training / symmetric-0.3 / thr=0.01 | 48.5% | -10.8% |
| PubMed | gcn | clean-training / symmetric-0.3 / thr=0.01 | 48.0% | clean-training / symmetric-0.3 / thr=0.01 | 48.2% | 0.2% |
| PubMed | graphsage | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / thr=0.05 | 72.9% | clean-training / symmetric-0.3-top2 / thr=0.01 | 73.8% | 0.9% |

### Target Coverage

| Dataset | Arch | Attack | Variant | Evaluated Targets | Goal |
|---|---|---|---|---:|---:|
| CiteSeer | gcn | adaptive-purified | clean-training | 9.000 | 20 |
| CiteSeer | gcn | adaptive-purified | clean-warmstart-symmetric-finetune | 9.600 | 20 |
| CiteSeer | gcn | standard | clean-training | 20.000 | 20 |
| CiteSeer | gcn | standard | clean-warmstart-symmetric-finetune | 20.000 | 20 |
| CiteSeer | graphsage | adaptive-purified | clean-training | 7.600 | 20 |
| CiteSeer | graphsage | adaptive-purified | clean-warmstart-symmetric-finetune | 7.200 | 20 |
| CiteSeer | graphsage | standard | clean-training | 16.800 | 20 |
| CiteSeer | graphsage | standard | clean-warmstart-symmetric-finetune | 16.800 | 20 |
| Cora | gcn | adaptive-purified | clean-training | 9.200 | 20 |
| Cora | gcn | adaptive-purified | clean-warmstart-symmetric-finetune | 9.200 | 20 |
| Cora | gcn | standard | clean-training | 19.400 | 20 |
| Cora | gcn | standard | clean-warmstart-symmetric-finetune | 19.400 | 20 |
| Cora | graphsage | adaptive-purified | clean-training | 7.200 | 20 |
| Cora | graphsage | adaptive-purified | clean-warmstart-symmetric-finetune | 7.200 | 20 |
| Cora | graphsage | standard | clean-training | 17.200 | 20 |
| Cora | graphsage | standard | clean-warmstart-symmetric-finetune | 17.200 | 20 |
| PubMed | gcn | adaptive-purified | clean-training | 19.400 | 20 |
| PubMed | gcn | adaptive-purified | clean-warmstart-symmetric-finetune | 19.200 | 20 |
| PubMed | gcn | standard | clean-training | 20.000 | 20 |
| PubMed | gcn | standard | clean-warmstart-symmetric-finetune | 20.000 | 20 |
| PubMed | graphsage | adaptive-purified | clean-training | 16.800 | 20 |
| PubMed | graphsage | adaptive-purified | clean-warmstart-symmetric-finetune | 16.800 | 20 |
| PubMed | graphsage | standard | clean-training | 17.000 | 20 |
| PubMed | graphsage | standard | clean-warmstart-symmetric-finetune | 17.000 | 20 |

### Final Figures

- [results/winner_only_benchmark_suite_v1/figures/winner_only_best_fixed_overview.png](results/winner_only_benchmark_suite_v1/figures/winner_only_best_fixed_overview.png)
- [results/winner_only_benchmark_suite_v1/figures/winner_only_selector_oracle_overview.png](results/winner_only_benchmark_suite_v1/figures/winner_only_selector_oracle_overview.png)
- [results/winner_only_benchmark_suite_v1/figures/winner_only_target_pool_coverage.png](results/winner_only_benchmark_suite_v1/figures/winner_only_target_pool_coverage.png)

## How To Read The Final Results

- Use [results/final_report.md](results/final_report.md) as the authoritative summary.

- Use [results/winner_only_benchmark_suite_v1/winner_only_fixed_config_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_fixed_config_summary.csv) when you want the deployable fixed-defense view across datasets.

- Use [results/winner_only_benchmark_suite_v1/winner_only_selector_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_selector_summary.csv) and [results/winner_only_benchmark_suite_v1/winner_only_oracle_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_oracle_summary.csv) when you want to compare deployable selection against the best-of-candidates upper envelope.

- Use [results/winner_only_benchmark_suite_v1/winner_only_target_pool_summary.csv](results/winner_only_benchmark_suite_v1/winner_only_target_pool_summary.csv) and [results/winner_only_benchmark_suite_v1/figures/winner_only_target_pool_coverage.png](results/winner_only_benchmark_suite_v1/figures/winner_only_target_pool_coverage.png) to interpret coverage limits in adaptive settings.

- Use the original Cora artifacts when you want the attack and repair case study material:
  - [results/attack_budget_accuracy.png](results/attack_budget_accuracy.png)
  - [results/purification_tradeoff.png](results/purification_tradeoff.png)
  - [results/nettack_target_purification_sweep.csv](results/nettack_target_purification_sweep.csv)
  - [results/preliminary_results.json](results/preliminary_results.json)

## Remaining Gaps And Limitations

- Target coverage is still incomplete in many adaptive winner-only combinations. The final suite is complete, but some combinations still fall below the intended 20-target goal.

- Several winner configurations achieve strong positive-certified fractions while still having mean radius near zero in the aggregate summary. Maximum radii can still be high, but average certificate strength is weaker than the best-case radius suggests.

- The low-noise asymmetric branch remains ablation-only evidence. The old `0.01` and `0.02` sparse asymmetric configurations still collapse to zero or near-zero certification under the current setup.

- Use [scripts/run_asymmetric_certificate_isolation.py](scripts/run_asymmetric_certificate_isolation.py) when you need to study the asymmetric collapse in more detail. It writes detailed and summary CSVs under [results/asymmetric_certificate_isolation_v1](results/asymmetric_certificate_isolation_v1).

- The clean and clean-warmstart symmetric fine-tuning variants remain the primary baselines to beat on the purified certificate benchmark. The matched sparse-noisy branch remains interesting, but it is not the current leader.