# Final Report

## Status

- Winner-only completed combinations: 12 / 12
- Winner-only datasets present: CiteSeer, Cora, PubMed
- Winner-only cross-dataset coverage is complete.

## Training Baselines

| Variant | Clean Test | Best Val | Global Attack @ 50 |
|---|---:|---:|---:|
| clean-training | 0.817 | 0.796 | 0.791 |
| matched-sparse-noisy-training | 0.817 | 0.806 | 0.770 |
| clean-warmstart-symmetric-finetune | 0.809 | 0.798 | 0.777 |

## Cora Mainline Best Results

| Variant | Best Fixed | Fixed Thr | Fixed Pos Cert | Fixed Correct | Best Oracle Thr | Oracle Pos Cert | Oracle Correct |
|---|---|---:|---:|---:|---:|---:|---:|
| balanced-sparse-noisy-training | symmetric-0.3 | 0.01 | 13.9% | 13.9% | 0.01 | 13.9% | 13.9% |
| clean-training | symmetric-0.3-top2 | 0.05 | 29.5% | 43.4% | 0.05 | 29.5% | 46.2% |
| clean-warmstart-symmetric-finetune | symmetric-0.3-top2 | 0.02 | 29.5% | 43.4% | 0.01 | 29.5% | 46.2% |
| matched-sparse-noisy-training | symmetric-0.3-top2 | 0.05 | 19.4% | 33.3% | 0.05 | 19.4% | 41.7% |
| purification-aware-symmetric-finetune | symmetric-0.3-cosine-top2 | 0.02 | 29.5% | 43.4% | 0.05 | 29.5% | 46.2% |

## Winner-Only Best Fixed Results

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

## Selector Vs Oracle

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

## Target Coverage

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

## Figures

- results\winner_only_benchmark_suite_v1\figures\winner_only_best_fixed_overview.png: present
- results\winner_only_benchmark_suite_v1\figures\winner_only_selector_oracle_overview.png: present
- results\winner_only_benchmark_suite_v1\figures\winner_only_target_pool_coverage.png: present
