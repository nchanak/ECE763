Install the dependencies in [requirements.txt](requirements.txt) before running the scripts.

Current Project State

- [main.py](main.py) runs the Cora robustness pipeline end to end: clean GCN training, matched sparse-noisy robust training, clean-warmstart symmetric fine-tuning, PRBCD global attack, Nettack targeted attack, Jaccard purification, smoothing, local certificates, and artifact generation.

- Attacks live in [src/attack.py](src/attack.py) and [src/nettack.py](src/nettack.py).

- Smoothing and certification logic lives in [src/smoothing.py](src/smoothing.py), with train/eval helpers in [src/train.py](src/train.py).

- Target-node Jaccard purification lives in [src/purification.py](src/purification.py).

- The post-purification certificate evaluation is split into two views:
  - Fixed-config evaluation in [results/purified_certificate_fixed_config_sweep.csv](results/purified_certificate_fixed_config_sweep.csv) and [results/purified_certificate_fixed_config_rows.csv](results/purified_certificate_fixed_config_rows.csv)
  - Oracle best-of-candidates exploration in [results/purified_certificate_oracle_sweep.csv](results/purified_certificate_oracle_sweep.csv) and [results/purified_certificate_oracle_rows.csv](results/purified_certificate_oracle_rows.csv)

- The purified certificate sweep now builds attack-aligned Nettack pools separately for the clean, matched sparse-noisy, and clean-warmstart symmetric fine-tuning models. Pool metadata is saved in [results/purified_certificate_target_pools.csv](results/purified_certificate_target_pools.csv).

- The pipeline can also emit a five-seed attack-aligned purified summary with `mean`, `sample_std`, `sem`, and `95%` confidence-interval columns in [results/purified_certificate_multiseed_oracle_summary.csv](results/purified_certificate_multiseed_oracle_summary.csv), [results/purified_certificate_multiseed_fixed_config_summary.csv](results/purified_certificate_multiseed_fixed_config_summary.csv), and the matching per-seed CSVs.

- The certificate-oriented warm-start variant writes its own training and global-attack artifacts to [results/certificate_oriented_training_history.csv](results/certificate_oriented_training_history.csv) and [results/certificate_oriented_attack_budget_accuracy.csv](results/certificate_oriented_attack_budget_accuracy.csv).

- The focus-node case study is now variant-specific. The selected nodes are saved in [results/focus_case_study_summary.csv](results/focus_case_study_summary.csv), and the purified focus sweeps are saved in [results/focus_purification_certificate_sweep.csv](results/focus_purification_certificate_sweep.csv).

- The legacy single-node focus diagnostics are also written per variant in [results/focus_variant_edge_drop_sweep.csv](results/focus_variant_edge_drop_sweep.csv), [results/focus_variant_local_certificate_sweep.csv](results/focus_variant_local_certificate_sweep.csv), and [results/focus_variant_sparse_local_certificate_sweep.csv](results/focus_variant_sparse_local_certificate_sweep.csv).

CLI And Configurable Runs

- [main.py](main.py) now accepts CLI configuration for dataset, seeds, output directory, training epochs, attack budgets, edge-drop sweep, sparse flip sweep, certificate node count, certificate radius, asymmetric delete/add limits, Nettack target count, and smoothing-mode selection.

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

- When multiple outer seeds are requested, [main.py](main.py) writes a top-level summary across those runs. The heavier purified five-seed attack-aligned summary remains the dedicated single-run artifact family listed above.

Latest Result Snapshot

- Latest confirmed single-seed summary: clean test accuracy 0.817, matched sparse-noisy test accuracy 0.811, clean-warmstart symmetric fine-tune test accuracy 0.814. The corresponding summary rows are written to [results/training_variant_summary.csv](results/training_variant_summary.csv) when the updated pipeline is rerun.

- Under PRBCD budget 50 in the latest confirmed single-seed validation, the clean model drops to 0.788 test accuracy, the matched sparse-noisy model drops to 0.764, and the clean-warmstart symmetric fine-tune model drops to 0.783. See [results/attack_budget_accuracy.csv](results/attack_budget_accuracy.csv), [results/robust_attack_budget_accuracy.csv](results/robust_attack_budget_accuracy.csv), and [results/certificate_oriented_attack_budget_accuracy.csv](results/certificate_oriented_attack_budget_accuracy.csv).

- Target-node Jaccard purification still recovers 4 of 5 sampled Nettack targets at thresholds 0.01 to 0.05. See [results/nettack_target_purification_sweep.csv](results/nettack_target_purification_sweep.csv).

- On the attack-aligned fixed-config purified sweep, the clean model only gets positive radii from the strong symmetric configuration, while the matched sparse-noisy model keeps better purified correctness with `sparse-asym-0.01` but gets zero positive certified cases on its own attacked target pool. See [results/purified_certificate_fixed_config_sweep.csv](results/purified_certificate_fixed_config_sweep.csv).

- On the oracle purified sweep, the clean model reaches 6/11 purified-correct targets with 1/11 positive certified targets at thresholds 0.01, 0.02, and 0.05. The matched sparse-noisy model reaches 7/11 purified-correct targets at thresholds 0.01 and 0.02, but 0/11 positive certified targets after switching to its own attack-aligned Nettack pool. See [results/purified_certificate_oracle_sweep.csv](results/purified_certificate_oracle_sweep.csv).

- The previous checked-in multiseed CSVs came from a shorter three-seed run. The updated code expands that summary to five seeds and records uncertainty columns directly in the aggregate CSVs so that comparisons are reported with explicit dispersion instead of just min/max spread.

How To Read The Purified Results

- Use [results/purified_certificate_fixed_config_sweep.csv](results/purified_certificate_fixed_config_sweep.csv) when you want a deployable fixed-defense view. Each row is one threshold and one fixed smoothing/certificate configuration.

- Use [results/purified_certificate_oracle_sweep.csv](results/purified_certificate_oracle_sweep.csv) only as an exploratory upper envelope. It picks the best correct candidate per target after evaluating the full candidate family.

- The compatibility files [results/purified_certificate_candidate_sweep.csv](results/purified_certificate_candidate_sweep.csv) and [results/purified_certificate_sweep.csv](results/purified_certificate_sweep.csv) still exist, but they mirror the clearer fixed-config and oracle views above.

Remaining Gaps

- The multiseed purified rerun is expensive because it evaluates three model variants across five seeds with confidence intervals. Regenerate those CSVs locally when you need updated aggregate numbers.

- Sparse asymmetric certificates remain the best way to preserve purified correctness, but they still do not yield positive radii on the current purified Nettack targets.

- The clean-warmstart symmetric fine-tuning variant improves the certificate-oriented single-seed tradeoff relative to the matched sparse-noisy model on the current validation run, but it still needs the full five-seed rerun before making a stronger claim.

- The saved CSV diagnostics are variant-specific, but the existing PNG plot outputs still visualize the clean focus node for continuity with the older report layout.

- The target attack surrogate is shared across variants through DeeProBUST Nettack, even though target selection and success measurement are now variant-specific.