# Final Report Draft

## Status

This draft separates validated aggregate results from partial or debug-only evidence.

Validated aggregate sources:

- `results/training_variant_summary.csv`
- `results/preliminary_results.json`
- `results/purified_mainline_multiseed_v4/purified_certificate_multiseed_fixed_config_summary.csv`
- `results/purified_mainline_multiseed_v4/purified_certificate_multiseed_oracle_summary.csv`
- `results/purified_mainline_multiseed_v4/purified_certificate_multiseed_ablation_summary.csv`
- `results/purified_mainline_multiseed_v4/purified_certificate_multiseed_target_pool_summary.csv`
- `results/winner_only_benchmark_suite_v1/winner_only_fixed_config_summary.csv`
- `results/winner_only_benchmark_suite_v1/winner_only_selector_summary.csv`
- `results/winner_only_benchmark_suite_v1/winner_only_oracle_summary.csv`
- `results/winner_only_benchmark_suite_v1/winner_only_target_pool_summary.csv`

Supplementary debug-only source:

- `results/debug_citeseer_graphsage_standard_seed7/winner_only_fixed_config_summary.csv`

Current completeness:

- The Cora purified mainline multiseed results are complete enough to support final claims.
- The winner-only suite root aggregates currently contain only Cora combinations.
- `results/winner_only_benchmark_suite_v1/winner_only_benchmark_suite_summary.md` is stale and should not be treated as the final suite summary.
- The `CiteSeer / graphsage / standard` branch has a successful single-seed debug run, but not a completed multiseed aggregate run.
- There are no completed PubMed artifacts under `results/winner_only_benchmark_suite_v1`.

## Executive Summary

1. On the validated Cora multiseed benchmark, the strongest deployable family remains the `symmetric-0.3` purification/certificate regime. For `clean-training` and `clean-warmstart-symmetric-finetune`, the best fixed configuration reaches `43.4%` correctness with `29.5%` positive certification and a mean max reported radius of `3.67`.
2. Oracle selection improves correctness more than certification. For the leading Cora variants, the oracle rises to `46.2%` correctness while the positive-certified fraction stays at `29.5%`, indicating that the main upside is picking the right config per node rather than expanding the certification ceiling.
3. `matched-sparse-noisy-training` remains a secondary branch. It keeps competitive clean accuracy but trails the leading symmetric family in certification, topping out at `19.4%` positive certification and `41.7%` oracle correctness.
4. `balanced-sparse-noisy-training` did not validate as the new mainline direction. Its best fixed and oracle aggregate point remains `13.9%` positive certification and `13.9%` correctness.
5. The legacy low-noise sparse-asymmetric settings (`0.01` and `0.02`) collapse to `0.0` positive certification across all tracked variants. The looser `legacy-sparse-asym-0.05` setting does recover positive certification, but it caps out at mean max radius `1.0`, so it behaves like a shallow radius-1 regime rather than a replacement for the symmetric family.
6. In the completed winner-only Cora benchmark, GraphSAGE is the clear leader. The best completed `graphsage / standard` result reaches `61.8%` correctness and `59.3%` positive certification, compared with `32.5%` and `24.4%` for the best completed `gcn / standard` result.
7. Adaptive purified attacks reduce certified performance on completed Cora runs. Relative to the best standard setting, the best adaptive setting is lower by `2.1` percentage points for GCN and `10.8` percentage points for GraphSAGE.
8. On the completed Cora winner-only runs, the selector already saturates the oracle envelope. The selector-oracle positive-certification gap is `0.0` in every completed combination.

## Cora Baselines

| Variant | Clean Test Acc | Best Val Acc | Global Attack Acc @ Budget 50 |
|---|---:|---:|---:|
| clean-training | 0.817 | 0.796 | 0.791 |
| matched-sparse-noisy-training | 0.817 | 0.806 | 0.770 |
| clean-warmstart-symmetric-finetune | 0.809 | 0.798 | 0.777 |

Interpretation:

- `matched-sparse-noisy-training` improves validation fit, but that gain does not translate into the strongest attack-budget-50 performance.
- `clean-warmstart-symmetric-finetune` gives up a small amount of clean accuracy for better certificate-oriented behavior later in the purified evaluation.

## Cora Mainline Purified Multiseed Results

### Best Fixed And Oracle Results By Variant

| Variant | Best Fixed Config | Fixed Thr | Fixed Pos Cert | Fixed Correct | Fixed Max Radius | Best Oracle Thr | Oracle Pos Cert | Oracle Correct | Nodes With Correct Config |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean-training | symmetric-0.3 | 0.01 | 0.2955 | 0.4343 | 3.6667 | 0.02 | 0.2955 | 0.4621 | 5.3333 |
| clean-warmstart-symmetric-finetune | symmetric-0.3 | 0.01 | 0.2955 | 0.4343 | 3.6667 | 0.01 | 0.2955 | 0.4621 | 5.3333 |
| matched-sparse-noisy-training | symmetric-0.3-top2 | 0.05 | 0.1944 | 0.3333 | 2.0000 | 0.05 | 0.1944 | 0.4167 | 5.0000 |
| balanced-sparse-noisy-training | balanced-sparse-0.2 | 0.01 | 0.1389 | 0.1389 | 1.3333 | 0.01 | 0.1389 | 0.1389 | 1.6667 |
| purification-aware-symmetric-finetune | symmetric-0.3 | 0.01 | 0.2955 | 0.4343 | 3.6667 | 0.01 | 0.2955 | 0.4621 | 5.3333 |

Main takeaways:

- The `symmetric-0.3` family remains the most reliable fixed choice for the strongest Cora variants.
- Oracle gains are mostly correctness gains. For the leading variants, the positive-certified fraction is already saturated by the fixed family, but the oracle can pick the right node-specific config often enough to move correctness from `43.4%` to `46.2%`.
- `balanced-sparse-noisy-training` is substantially weaker than the clean and warm-started symmetric families and should not be treated as the main recommendation.

### Attack-Pool Coverage

| Variant | Target Goal | Successful Attacks Mean | Evaluated Targets Mean |
|---|---:|---:|---:|
| balanced-sparse-noisy-training | 12 | 12.000 | 12.000 |
| clean-training | 12 | 11.667 | 11.667 |
| clean-warmstart-symmetric-finetune | 12 | 11.667 | 11.667 |
| matched-sparse-noisy-training | 12 | 12.000 | 12.000 |
| purification-aware-symmetric-finetune | 12 | 11.667 | 11.667 |

Coverage is effectively complete for the Cora multiseed attack-aligned pool. The aggregate numbers above are not being driven by severe target-count collapse.

## Legacy Sparse-Asymmetric Ablation

### Headline Tradeoff

| Variant | Best Low-Noise Asym Pos Cert (`0.01`/`0.02`) | Best Legacy `0.05` Pos Cert |
|---|---:|---:|
| clean-training | 0.0000 | 0.3687 |
| clean-warmstart-symmetric-finetune | 0.0000 | 0.2525 |
| matched-sparse-noisy-training | 0.0000 | 0.2778 |
| balanced-sparse-noisy-training | 0.0000 | 0.2778 |
| purification-aware-symmetric-finetune | 0.0000 | 0.2854 |

### Best Legacy `0.05` Rows By Variant

| Variant | Config | Thr | Pos Cert | Correct | Abstain | Correct Zero Radius | Max Radius |
|---|---|---:|---:|---:|---:|---:|---:|
| clean-training | legacy-sparse-asym-0.05 | 0.01 | 0.3687 | 0.6616 | 0.0000 | 0.2929 | 1.0000 |
| clean-warmstart-symmetric-finetune | legacy-sparse-asym-0.05 | 0.01 | 0.2525 | 0.5707 | 0.0000 | 0.3182 | 1.0000 |
| matched-sparse-noisy-training | legacy-sparse-asym-0.05 | 0.01 | 0.2778 | 0.6111 | 0.0000 | 0.3333 | 1.0000 |
| balanced-sparse-noisy-training | legacy-sparse-asym-0.05 | 0.02 | 0.2778 | 0.6667 | 0.0000 | 0.3889 | 1.0000 |
| purification-aware-symmetric-finetune | legacy-sparse-asym-0.05 | 0.01 | 0.2854 | 0.6616 | 0.0000 | 0.3763 | 1.0000 |

Interpretation:

- The low-noise asymmetric branch is not viable. Across all tracked variants, the `0.01` and `0.02` legacy sparse-asymmetric settings collapse to zero positive certification.
- The looser `legacy-sparse-asym-0.05` branch does recover non-zero certification and, on some variants, very high correctness. However, every best legacy row caps out at mean max radius `1.0`.
- The current evidence therefore supports a tradeoff interpretation: the legacy asymmetric branch can recover many shallow certificates, while the symmetric family remains the better candidate when broader-radius behavior is part of the objective.

## Winner-Only Benchmark Suite Snapshot

The root winner-only suite is currently a Cora-only aggregate snapshot.

- Root fixed rows: `72`
- Root selector rows: `24`
- Root oracle rows: `24`
- Root target-pool rows: `8`
- Root datasets present: `Cora` only

### Best Completed Cora Winners

| Dataset | Arch | Attack | Winner Family | Thr | Pos Cert | Correct | Abstain | Max Radius |
|---|---|---|---|---:|---:|---:|---:|---:|
| Cora | gcn | standard | clean-training / symmetric-0.3-cosine-top2 | 0.01 | 0.2443 | 0.3253 | 0.0705 | 4.0000 |
| Cora | gcn | adaptive-purified | symmetric-0.3 family | 0.02 | 0.2237 | 0.2964 | 0.0545 | 2.4000 |
| Cora | graphsage | standard | clean-training / symmetric-0.3 | 0.02 | 0.5931 | 0.6181 | 0.0125 | 5.0000 |
| Cora | graphsage | adaptive-purified | clean-training / symmetric-0.3 | 0.01 | 0.4855 | 0.4855 | 0.0000 | 4.0000 |

Main takeaways:

- On completed Cora winner-only evidence, GraphSAGE is far ahead of GCN.
- The best `graphsage / standard` result improves on the best `gcn / standard` result by `34.9` percentage points in positive certification and `29.3` percentage points in correctness.
- The best completed adaptive runs are still competitive, but adaptive purified attacks clearly reduce certification headroom, especially for GraphSAGE.

### Selector Vs Oracle On Completed Cora Runs

| Dataset | Arch | Attack | Selector Thr | Selector Pos Cert | Oracle Pos Cert | Gap | Selector Correct | Oracle Correct |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Cora | gcn | standard | 0.01 | 0.2443 | 0.2443 | 0.0000 | 0.3248 | 0.3353 |
| Cora | gcn | adaptive-purified | 0.05 | 0.2237 | 0.2237 | 0.0000 | 0.2964 | 0.2964 |
| Cora | graphsage | standard | 0.02 | 0.5931 | 0.5931 | 0.0000 | 0.6181 | 0.6181 |
| Cora | graphsage | adaptive-purified | 0.01 | 0.4855 | 0.4855 | 0.0000 | 0.4855 | 0.4855 |

Interpretation:

- On the completed Cora combinations, the selector is already at the oracle certification ceiling.
- The only visible oracle lift is a small correctness gain for `Cora / gcn / standard`.

### Standard Vs Adaptive On Completed Cora Runs

| Arch | Best Standard Pos Cert | Best Adaptive Pos Cert | Delta |
|---|---:|---:|---:|
| gcn | 0.2443 | 0.2237 | -0.0206 |
| graphsage | 0.5931 | 0.4855 | -0.1076 |

Adaptive purified attacks reduce the strongest completed Cora certification rates, and the drop is much larger for GraphSAGE than for GCN.

## Partial And Debug-Only Evidence

### CiteSeer / GraphSAGE / Standard Single-Seed Debug Run

The single-seed debug run under `results/debug_citeseer_graphsage_standard_seed7` succeeded and gives a useful lower-risk signal for the unfinished branch.

Best observed debug row:

- model variant: `clean-training`
- config: `symmetric-0.3`
- threshold: `0.01`
- positive certification: `0.3750`
- correctness: `0.5000`
- abstention: `0.0625`
- mean max reported radius: `5.0000`

This is useful evidence that the `CiteSeer / graphsage / standard` path is not fundamentally broken at single-seed scale. It is not, however, a substitute for the missing multiseed aggregate.

### Missing Cross-Dataset Coverage

- The root winner-only aggregate CSVs do not yet include any completed CiteSeer or PubMed combinations.
- The root markdown summary is therefore a stale partial snapshot, not a final cross-dataset report.

## Claims That Are Safe To Carry Forward

1. The `symmetric-0.3` family is the strongest validated Cora mainline certificate family.
2. `clean-training` and `clean-warmstart-symmetric-finetune` remain the main baselines to beat on Cora.
3. `matched-sparse-noisy-training` is still interesting, but it is not the current leader.
4. `balanced-sparse-noisy-training` should not be treated as the new mainline direction.
5. Low-noise sparse-asymmetric certification is not viable in the current setup.
6. The looser sparse-asymmetric regime mostly recovers shallow radius-1 certificates rather than broader-radius certificates.
7. On completed cross-dataset-style evidence, GraphSAGE is the strongest architecture currently validated.
8. Adaptive purified attacks materially reduce certification on completed Cora runs.
9. The selector leaves essentially no measurable oracle certification headroom on the completed Cora winner-only combinations.

## Remaining Work Before Calling The Report Final

1. Finish the missing winner-only multiseed combinations outside Cora, especially `CiteSeer / graphsage / standard`.
2. Capture the actual traceback for the remaining multiseed GraphSAGE failure in the full-suite context.
3. Complete or rerun the PubMed branch so the root suite aggregates stop being Cora-only.
4. Regenerate `results/winner_only_benchmark_suite_v1/winner_only_benchmark_suite_summary.md` only after the root CSVs contain all intended combinations.