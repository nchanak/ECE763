# Winner-Only Benchmark Suite Summary

Output directory: C:\College\ECE763\results\winner_only_graphsage_smoke_v1

## Key Findings

- Best fixed configuration overall: Cora / graphsage / standard / clean-training / symmetric-0.3 / thr=0.01 with positive certification 0.0%, correctness 0.0%, and mean radius 0.000.
- Best selector summary overall: Cora / graphsage / standard / thr=0.01 at positive certification 0.0%.
- Oracle headroom overall: best oracle reaches 0.0%, a 0.0% lift over the best selector point.

## Fixed-Config Winners

| Dataset | Arch | Attack | Winner | Positive Cert | Correct | Mean Radius | Max Radius | Abstain |
|---|---|---|---|---:|---:|---:|---:|---:|
| Cora | graphsage | standard | clean-training / symmetric-0.3 / jaccard / rest / thr=0.01 | 0.0% | 0.0% | 0.000 | 0.000 | 0.0% |

## Selector Vs Oracle

| Dataset | Arch | Attack | Selector Threshold | Selector Positive Cert | Oracle Positive Cert | Gap | Selector Correct | Oracle Correct |
|---|---|---|---|---:|---:|---:|---:|---:|
| Cora | graphsage | standard | 0.01 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Standard Vs Adaptive

| Dataset | Arch | Best Standard Winner | Positive Cert | Best Adaptive Winner | Positive Cert | Delta |
|---|---|---|---:|---|---:|---:|
| No paired standard/adaptive results available yet | - | - | - | - | - | - |

## Attack Pool Coverage

| Dataset | Arch | Attack | Variant | Pool Mode | Successful Attacks | Evaluated Targets | Goal |
|---|---|---|---|---|---:|---:|---:|
| Cora | graphsage | standard | clean-training | successful-attacks | 1.000 | 1.000 | 1 |
| Cora | graphsage | standard | clean-warmstart-symmetric-finetune | successful-attacks | 1.000 | 1.000 | 1 |

## Failure Modes

- No dominant failure mode crossed the configured thresholds in the aggregate summaries.
