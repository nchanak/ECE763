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

- The pipeline can also emit a reduced three-seed attack-aligned purified summary for the current mainline-vs-ablation setup. The latest checked run is written under [results/purified_mainline_multiseed_v3/purified_certificate_multiseed_oracle_summary.csv](results/purified_mainline_multiseed_v3/purified_certificate_multiseed_oracle_summary.csv), [results/purified_mainline_multiseed_v3/purified_certificate_multiseed_fixed_config_summary.csv](results/purified_mainline_multiseed_v3/purified_certificate_multiseed_fixed_config_summary.csv), [results/purified_mainline_multiseed_v3/purified_certificate_multiseed_ablation_summary.csv](results/purified_mainline_multiseed_v3/purified_certificate_multiseed_ablation_summary.csv), and the matching per-seed CSVs.

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

- Latest confirmed purified multiseed summary: the mainline certificate family is now balanced-sparse-0.2 plus the nearby symmetric configs, with the legacy low-noise asymmetric family retained only as an ablation branch. The latest checked run is in [results/purified_mainline_multiseed_v3](results/purified_mainline_multiseed_v3).

- On the oracle purified multiseed summary, `clean-training` and `clean-warmstart-symmetric-finetune` are still the strongest training variants. They reach about `0.434-0.462` mean correct fraction and `0.268-0.295` mean positive-certified fraction depending on threshold, with roughly `5.0-5.33` nodes per seed having at least one correct certifiable config. See [results/purified_mainline_multiseed_v3/purified_certificate_multiseed_oracle_summary.csv](results/purified_mainline_multiseed_v3/purified_certificate_multiseed_oracle_summary.csv).

- `matched-sparse-noisy-training` remains a secondary robustness-alignment branch rather than the mainline leader. In the same oracle summary it reaches about `0.389-0.417` mean correct fraction and `0.167-0.194` mean positive-certified fraction.

- `balanced-sparse-noisy-training` did not validate as the new aligned-training direction. In the oracle summary it stays at `0.1389` mean correct fraction and `0.1389` mean positive-certified fraction at all thresholds, with only `1.67` nodes per seed finding a correct certifiable config.

- On the fixed-config summary, the best deployable mainline configs are still in the revised family rather than the legacy asymmetric family. `symmetric-0.3` is the most consistently strong fixed config for the clean and warm-started models, while `balanced-sparse-0.2` remains a reasonable fixed baseline inside the same family. See [results/purified_mainline_multiseed_v3/purified_certificate_multiseed_fixed_config_summary.csv](results/purified_mainline_multiseed_v3/purified_certificate_multiseed_fixed_config_summary.csv).

- In the ablation summary, the legacy low-noise asymmetric configs at `0.01` and `0.02` still collapse to zero positive-certified fraction across variants, which is why they remain ablation-only evidence. The looser `0.05` legacy asymmetric config can recover some positive certification, but it does not change the main direction. See [results/purified_mainline_multiseed_v3/purified_certificate_multiseed_ablation_summary.csv](results/purified_mainline_multiseed_v3/purified_certificate_multiseed_ablation_summary.csv).

How To Read The Purified Results

- Use [results/purified_certificate_fixed_config_sweep.csv](results/purified_certificate_fixed_config_sweep.csv) when you want a deployable fixed-defense view. Each row is one threshold and one fixed smoothing/certificate configuration.

- Use [results/purified_certificate_oracle_sweep.csv](results/purified_certificate_oracle_sweep.csv) only as an exploratory upper envelope. It picks the best correct candidate per target after evaluating the full candidate family.

- The compatibility files [results/purified_certificate_candidate_sweep.csv](results/purified_certificate_candidate_sweep.csv) and [results/purified_certificate_sweep.csv](results/purified_certificate_sweep.csv) still exist, but they mirror the clearer fixed-config and oracle views above.

Remaining Gaps

- The multiseed purified rerun is still expensive because it evaluates multiple model variants and certificate configs across a reduced three-seed attack-aligned sweep. Regenerate those CSVs locally when you need updated aggregate numbers.

- The active asymmetric certificate sweep is now decoupled from the mainline purified grid. The old `0.01` and `0.02` sparse asymmetric configs are retained only for diagnostics because they still collapse to zero-radius in the multiseed purified ablation summary.

- Use [scripts/run_asymmetric_certificate_isolation.py](scripts/run_asymmetric_certificate_isolation.py) to isolate three competing explanations for zero-radius collapse on purified targets: the strict `p_rest_upper` bound, the delete/add budget search itself, and the sparse candidate grid. The script writes detailed and summary CSVs under `results/asymmetric_certificate_isolation_v1` by default.

- The clean and clean-warmstart symmetric fine-tuning variants are the primary baselines to beat on the current purified certificate benchmark. The matched sparse-noisy variant remains worth iterating, but balanced-sparse-noisy-training should be treated as secondary until its training noise is redesigned.

- The saved CSV diagnostics are variant-specific, but the existing PNG plot outputs still visualize the clean focus node for continuity with the older report layout.

- The target attack surrogate is shared across variants through DeeProBUST Nettack, even though target selection and success measurement are now variant-specific.