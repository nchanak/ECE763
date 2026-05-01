# Winner-Only Benchmark Suite Summary

Output directory: C:\College\ECE763\results\winner_only_benchmark_suite_v1

## Key Findings

- Best fixed configuration overall: PubMed / graphsage / adaptive-purified / clean-training / symmetric-0.3-top2 / thr=0.01 with positive certification 73.8%, correctness 77.2%, and mean radius 0.000.
- Best selector summary overall: PubMed / graphsage / adaptive-purified / thr=0.02 at positive certification 73.8%.
- Oracle headroom overall: best oracle reaches 73.8%, a 0.0% lift over the best selector point.

## Fixed-Config Winners

| Dataset | Arch | Attack | Winner | Positive Cert | Correct | Mean Radius | Max Radius | Abstain |
|---|---|---|---|---:|---:|---:|---:|---:|
| CiteSeer | gcn | adaptive-purified | clean-training / symmetric-0.3 / jaccard / rest / thr=0.01 | 17.9% | 20.2% | 0.000 | 4.000 | 2.2% |
| CiteSeer | gcn | standard | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / cosine / top2 / thr=0.02 | 18.0% | 22.0% | 0.000 | 4.600 | 5.0% |
| CiteSeer | graphsage | adaptive-purified | clean-training / symmetric-0.3 / jaccard / rest / thr=0.01 | 43.2% | 46.0% | 0.000 | 5.000 | 0.0% |
| CiteSeer | graphsage | standard | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / cosine / top2 / thr=0.05 | 56.9% | 59.4% | 0.000 | 5.000 | 3.6% |
| Cora | gcn | adaptive-purified | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / jaccard / top2 / thr=0.05 | 22.4% | 29.6% | 0.000 | 2.400 | 3.6% |
| Cora | gcn | standard | clean-training / symmetric-0.3-cosine-top2 / cosine / top2 / thr=0.01 | 24.4% | 32.5% | 0.000 | 4.000 | 7.1% |
| Cora | graphsage | adaptive-purified | clean-training / symmetric-0.3 / jaccard / rest / thr=0.01 | 48.5% | 48.5% | 0.000 | 4.000 | 0.0% |
| Cora | graphsage | standard | clean-training / symmetric-0.3 / jaccard / rest / thr=0.02 | 59.3% | 61.8% | 0.000 | 5.000 | 1.2% |
| PubMed | gcn | adaptive-purified | clean-training / symmetric-0.3 / jaccard / rest / thr=0.01 | 48.2% | 48.2% | 0.000 | 5.000 | 0.0% |
| PubMed | gcn | standard | clean-training / symmetric-0.3 / jaccard / rest / thr=0.01 | 48.0% | 48.0% | 0.000 | 5.000 | 0.0% |
| PubMed | graphsage | adaptive-purified | clean-training / symmetric-0.3-top2 / jaccard / top2 / thr=0.01 | 73.8% | 77.2% | 0.000 | 5.000 | 2.2% |
| PubMed | graphsage | standard | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / jaccard / top2 / thr=0.05 | 72.9% | 76.4% | 0.000 | 5.000 | 1.2% |

## Selector Vs Oracle

| Dataset | Arch | Attack | Selector Threshold | Selector Positive Cert | Oracle Positive Cert | Gap | Selector Correct | Oracle Correct |
|---|---|---|---|---:|---:|---:|---:|---:|
| CiteSeer | gcn | adaptive-purified | 0.05 | 17.9% | 17.9% | 0.0% | 20.2% | 20.2% |
| CiteSeer | gcn | standard | 0.02 | 18.0% | 18.0% | 0.0% | 21.0% | 22.0% |
| CiteSeer | graphsage | adaptive-purified | 0.02 | 43.2% | 43.2% | 0.0% | 46.0% | 46.0% |
| CiteSeer | graphsage | standard | 0.01 | 55.7% | 56.9% | 1.2% | 59.4% | 59.4% |
| Cora | gcn | adaptive-purified | 0.05 | 22.4% | 22.4% | 0.0% | 29.6% | 29.6% |
| Cora | gcn | standard | 0.01 | 24.4% | 24.4% | 0.0% | 32.5% | 33.5% |
| Cora | graphsage | adaptive-purified | 0.01 | 48.5% | 48.5% | 0.0% | 48.5% | 48.5% |
| Cora | graphsage | standard | 0.02 | 59.3% | 59.3% | 0.0% | 61.8% | 61.8% |
| PubMed | gcn | adaptive-purified | 0.01 | 48.2% | 48.2% | 0.0% | 48.2% | 48.2% |
| PubMed | gcn | standard | 0.01 | 48.0% | 48.0% | 0.0% | 48.0% | 48.0% |
| PubMed | graphsage | adaptive-purified | 0.02 | 73.8% | 73.8% | 0.0% | 77.2% | 77.2% |
| PubMed | graphsage | standard | 0.01 | 72.9% | 72.9% | 0.0% | 76.4% | 76.4% |

## Standard Vs Adaptive

| Dataset | Arch | Best Standard Winner | Positive Cert | Best Adaptive Winner | Positive Cert | Delta |
|---|---|---|---:|---|---:|---:|
| CiteSeer | gcn | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / thr=0.02 | 18.0% | clean-training / symmetric-0.3 / thr=0.01 | 17.9% | -0.1% |
| CiteSeer | graphsage | clean-warmstart-symmetric-finetune / symmetric-0.3-cosine-top2 / thr=0.05 | 56.9% | clean-training / symmetric-0.3 / thr=0.01 | 43.2% | -13.7% |
| Cora | gcn | clean-training / symmetric-0.3-cosine-top2 / thr=0.01 | 24.4% | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / thr=0.05 | 22.4% | -2.1% |
| Cora | graphsage | clean-training / symmetric-0.3 / thr=0.02 | 59.3% | clean-training / symmetric-0.3 / thr=0.01 | 48.5% | -10.8% |
| PubMed | gcn | clean-training / symmetric-0.3 / thr=0.01 | 48.0% | clean-training / symmetric-0.3 / thr=0.01 | 48.2% | 0.2% |
| PubMed | graphsage | clean-warmstart-symmetric-finetune / symmetric-0.3-top2 / thr=0.05 | 72.9% | clean-training / symmetric-0.3-top2 / thr=0.01 | 73.8% | 0.9% |

## Attack Pool Coverage

| Dataset | Arch | Attack | Variant | Pool Mode | Successful Attacks | Evaluated Targets | Goal |
|---|---|---|---|---|---:|---:|---:|
| CiteSeer | gcn | adaptive-purified | clean-training | successful-attacks | 9.000 | 9.000 | 20 |
| CiteSeer | gcn | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 9.600 | 9.600 | 20 |
| CiteSeer | gcn | standard | clean-training | successful-attacks | 20.000 | 20.000 | 20 |
| CiteSeer | gcn | standard | clean-warmstart-symmetric-finetune | successful-attacks | 20.000 | 20.000 | 20 |
| CiteSeer | graphsage | adaptive-purified | clean-training | successful-attacks | 7.600 | 7.600 | 20 |
| CiteSeer | graphsage | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 7.200 | 7.200 | 20 |
| CiteSeer | graphsage | standard | clean-training | successful-attacks | 16.800 | 16.800 | 20 |
| CiteSeer | graphsage | standard | clean-warmstart-symmetric-finetune | successful-attacks | 16.800 | 16.800 | 20 |
| Cora | gcn | adaptive-purified | clean-training | successful-attacks | 9.200 | 9.200 | 20 |
| Cora | gcn | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 9.200 | 9.200 | 20 |
| Cora | gcn | standard | clean-training | successful-attacks | 19.400 | 19.400 | 20 |
| Cora | gcn | standard | clean-warmstart-symmetric-finetune | successful-attacks | 19.400 | 19.400 | 20 |
| Cora | graphsage | adaptive-purified | clean-training | successful-attacks | 7.200 | 7.200 | 20 |
| Cora | graphsage | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 7.200 | 7.200 | 20 |
| Cora | graphsage | standard | clean-training | successful-attacks | 17.200 | 17.200 | 20 |
| Cora | graphsage | standard | clean-warmstart-symmetric-finetune | successful-attacks | 17.200 | 17.200 | 20 |
| PubMed | gcn | adaptive-purified | clean-training | successful-attacks | 19.400 | 19.400 | 20 |
| PubMed | gcn | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 19.200 | 19.200 | 20 |
| PubMed | gcn | standard | clean-training | successful-attacks | 20.000 | 20.000 | 20 |
| PubMed | gcn | standard | clean-warmstart-symmetric-finetune | successful-attacks | 20.000 | 20.000 | 20 |
| PubMed | graphsage | adaptive-purified | clean-training | successful-attacks | 16.800 | 16.800 | 20 |
| PubMed | graphsage | adaptive-purified | clean-warmstart-symmetric-finetune | successful-attacks | 16.800 | 16.800 | 20 |
| PubMed | graphsage | standard | clean-training | successful-attacks | 17.000 | 17.000 | 20 |
| PubMed | graphsage | standard | clean-warmstart-symmetric-finetune | successful-attacks | 17.000 | 17.000 | 20 |

## Failure Modes

- Target coverage is incomplete in 20 combinations: evaluated targets came in below the requested goal.
