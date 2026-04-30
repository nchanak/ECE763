# Winner-Only Benchmark Suite Summary

Output directory: C:\College\ECE763\results\winner_only_benchmark_suite_v1

## Key Findings

- Best fixed configuration overall: Cora / graphsage / standard / clean-training / symmetric-0.3 / thr=0.02 with positive certification 59.3%, correctness 61.8%, and mean radius 0.000.
- Best selector summary overall: Cora / graphsage / standard / thr=0.02 at positive certification 59.3%.
- Oracle headroom overall: best oracle reaches 59.3%, a 0.0% lift over the best selector point.

## Fixed-Config Winners

| Dataset | Arch | Attack | Winner | Positive Cert | Correct | Mean Radius | Max Radius | Abstain |
|---|---|---|---|---:|---:|---:|---:|---:|
| Cora | gcn | adaptive-purified | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / jaccard / top2 / thr=0.05 | 22.4% | 29.6% | 0.000 | 2.400 | 3.6% |
| Cora | gcn | standard | clean-training / symmetric-0.3-cosine-top2 / cosine / top2 / thr=0.01 | 24.4% | 32.5% | 0.000 | 4.000 | 7.1% |
| Cora | graphsage | adaptive-purified | clean-training / symmetric-0.3 / jaccard / rest / thr=0.01 | 48.5% | 48.5% | 0.000 | 4.000 | 0.0% |
| Cora | graphsage | standard | clean-training / symmetric-0.3 / jaccard / rest / thr=0.02 | 59.3% | 61.8% | 0.000 | 5.000 | 1.2% |

## Selector Vs Oracle

| Dataset | Arch | Attack | Selector Threshold | Selector Positive Cert | Oracle Positive Cert | Gap | Selector Correct | Oracle Correct |
|---|---|---|---|---:|---:|---:|---:|---:|
| Cora | gcn | adaptive-purified | 0.05 | 22.4% | 22.4% | 0.0% | 29.6% | 29.6% |
| Cora | gcn | standard | 0.01 | 24.4% | 24.4% | 0.0% | 32.5% | 33.5% |
| Cora | graphsage | adaptive-purified | 0.01 | 48.5% | 48.5% | 0.0% | 48.5% | 48.5% |
| Cora | graphsage | standard | 0.02 | 59.3% | 59.3% | 0.0% | 61.8% | 61.8% |

## Standard Vs Adaptive

| Dataset | Arch | Best Standard Winner | Positive Cert | Best Adaptive Winner | Positive Cert | Delta |
|---|---|---|---:|---|---:|---:|
| Cora | gcn | clean-training / symmetric-0.3-cosine-top2 / thr=0.01 | 24.4% | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / thr=0.05 | 22.4% | -2.1% |
| Cora | graphsage | clean-training / symmetric-0.3 / thr=0.02 | 59.3% | clean-training / symmetric-0.3 / thr=0.01 | 48.5% | -10.8% |

## Attack Pool Coverage

| Dataset | Arch | Attack | Variant | Pool Mode | Successful Attacks | Evaluated Targets | Goal |
|---|---|---|---|---|---:|---:|---:|
| Cora | gcn | adaptive-purified | clean-training | successful-attacks | 9.200 | 9.200 | 20 |
| Cora | gcn | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 9.200 | 9.200 | 20 |
| Cora | gcn | standard | clean-training | successful-attacks | 19.400 | 19.400 | 20 |
| Cora | gcn | standard | clean-warmstart-symmetric-finetune | successful-attacks | 19.400 | 19.400 | 20 |
| Cora | graphsage | adaptive-purified | clean-training | successful-attacks | 7.200 | 7.200 | 20 |
| Cora | graphsage | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 7.200 | 7.200 | 20 |
| Cora | graphsage | standard | clean-training | successful-attacks | 17.200 | 17.200 | 20 |
| Cora | graphsage | standard | clean-warmstart-symmetric-finetune | successful-attacks | 17.200 | 17.200 | 20 |

## Failure Modes

- Target coverage is incomplete in 8 combinations: evaluated targets came in below the requested goal.
