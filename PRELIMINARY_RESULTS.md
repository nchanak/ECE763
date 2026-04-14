# Preliminary Results

Generated from the current Cora experiment pipeline in `main.py` using the validated `.venv` environment.

## Setup

- Dataset: Cora
- Graph size: 2708 nodes, 5278 undirected edges
- Model: 2-layer PyG GCN
- Environment: Python 3.11.9, torch 2.11.0+cpu, torch-geometric 2.7.0, torchaudio 2.11.0+cpu, scipy 1.15.3
- Main run command: `c:/College/ECE763/.venv/Scripts/python.exe main.py`

## Headline Takeaways

- The base model is competitive on clean Cora: best validation accuracy reached `0.796`, best test accuracy reached `0.817`, and the final clean evaluation reported `0.806` test accuracy.
- PRBCD degrades test accuracy as the global perturbation budget increases, reaching `0.764` at a 50-edge budget.
- Nettack remains effective on sampled targets: `3/5` sampled target nodes flipped to an incorrect class under the current targeted-attack setup.
- The smoothing implementations preserve overall clean accuracy reasonably well, but the current local symmetric certificate is still very weak on the small evaluated target-node subset.

## Training Snapshot

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 0.796 |
| Best test accuracy | 0.817 |
| Final clean train accuracy | 1.000 |
| Final clean validation accuracy | 0.790 |
| Final clean test accuracy | 0.806 |

Training history is saved in [results/training_history.csv](results/training_history.csv).

## Global Structural Attack

PRBCD was evaluated at increasing perturbation budgets.

| PRBCD budget | Test accuracy |
| --- | ---: |
| 5 | 0.803 |
| 10 | 0.795 |
| 20 | 0.790 |
| 50 | 0.764 |

The drop from the clean test accuracy (`0.806`) to budget 50 (`0.764`) is `0.042` absolute.

Artifacts:

- [results/attack_budget_accuracy.csv](results/attack_budget_accuracy.csv)
- [results/attack_budget_accuracy.png](results/attack_budget_accuracy.png)

## Targeted Attack Snapshot

Five Nettack target nodes were sampled for a quick case study.

| Target node | True label | Clean pred | Attacked pred | Attack success |
| --- | ---: | ---: | ---: | --- |
| 1709 | 2 | 2 | 2 | No |
| 1710 | 2 | 2 | 3 | Yes |
| 1711 | 2 | 2 | 2 | No |
| 1712 | 2 | 2 | 0 | Yes |
| 1713 | 0 | 0 | 2 | Yes |

This is enough to show that targeted structural perturbations are still effective against the current base classifier.

## Smoothing Baselines

### Edge-drop smoothing

The edge-drop baseline stayed close to the clean model in aggregate test accuracy.

| `p_delete` | Clean test acc | Attacked test acc |
| --- | ---: | ---: |
| 0.05 | 0.806 | 0.805 |
| 0.10 | 0.805 | 0.804 |
| 0.20 | 0.806 | 0.806 |
| 0.30 | 0.807 | 0.805 |
| 0.40 | 0.802 | 0.801 |

The baseline edge-drop summary reported in the JSON is:

- Train: `1.000`
- Val: `0.790`
- Test: `0.805`

Artifacts:

- [results/edge_drop_sweep.csv](results/edge_drop_sweep.csv)
- [results/edge_drop_tradeoff.png](results/edge_drop_tradeoff.png)

### Sparse edge-flip smoothing

Sparse edge-flip smoothing also preserved clean accuracy well on the current run.

| `p_delete` | `p_add` | `max_additions` | Clean test acc | Attacked test acc |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 5e-06 | 64 | 0.806 | 0.805 |
| 0.02 | 1e-05 | 96 | 0.806 | 0.805 |
| 0.05 | 2e-05 | 160 | 0.806 | 0.805 |

The current tuned sparse-flip baseline is:

- Test accuracy: `0.806`
- `p_delete = 0.02`
- `p_add = 1e-5`
- `max_additions = 96`

Artifacts:

- [results/sparse_edge_flip_sweep.csv](results/sparse_edge_flip_sweep.csv)
- [results/sparse_flip_tradeoff.png](results/sparse_flip_tradeoff.png)

## Local Certificate Results

### Target-node certificate sweep

The selected focus node was `1712` with true label `2`. Under the current local symmetric edge-flip smoothing setup:

- Clean probe correct: `False`
- Attacked probe correct: `False`
- Clean reported radius: `0`
- Attacked reported radius: `0`

| `p_flip` | `beta` | Clean radius | Attacked radius | Clean `pA_lower` | Attacked `pA_lower` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.99 | 0 | 0 | 0.4122 | 0.2924 |
| 0.02 | 0.98 | 0 | 0 | 0.6227 | 0.4516 |
| 0.05 | 0.95 | 0 | 0 | 0.6491 | 0.6122 |
| 0.10 | 0.90 | 0 | 0 | 0.7002 | 0.6598 |

Artifacts:

- [results/local_certificate_sweep.csv](results/local_certificate_sweep.csv)
- [results/local_certificate_radius.png](results/local_certificate_radius.png)

### Certified-accuracy subset sweep

The local certified-accuracy experiment currently evaluates only 10 test nodes: `1709` through `1718`.

| `p_flip` | Evaluated nodes | Correct fraction | Positive certified accuracy | Mean certified radius on correct |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 10 | 0.20 | 0.00 | 0.00 |
| 0.02 | 10 | 0.00 | 0.00 | 0.00 |
| 0.05 | 10 | 0.00 | 0.00 | 0.00 |
| 0.10 | 10 | 0.00 | 0.00 | 0.00 |

At the moment, the best observed certified accuracy is only `0.20` at radius `0` for `p_flip = 0.01`, and `0.00` for radius `>= 1` across the current sweep.

Artifacts:

- [results/symmetric_certificate_sweep.csv](results/symmetric_certificate_sweep.csv)
- [results/certificate_subset_rows.csv](results/certificate_subset_rows.csv)
- [results/certified_accuracy_curve.csv](results/certified_accuracy_curve.csv)
- [results/certified_accuracy_curve.png](results/certified_accuracy_curve.png)

## Interpretation

- The core pipeline is working end to end: train, attack, smooth, certify, and report.
- Global robustness is limited, which is expected for a standard GCN on Cora under structural attack.
- The sparse smoothing machinery does not noticeably hurt clean accuracy in this preliminary run, which is useful for follow-up experiments.
- The current local symmetric certificate is too pessimistic to claim meaningful robustness yet. That is consistent with the motivation for moving toward a sparsity-aware asymmetric certificate.

## Caveats

- These numbers are preliminary and come from the current implementation state, not a large multi-seed study.
- The local certificate analysis is only over a small 10-node subset.
- The certification experiment uses symmetric local edge-flip noise, which is not yet the stronger sparse-aware formulation discussed in the proposal.

## Full Artifact Index

- [results/preliminary_results.json](results/preliminary_results.json)
- [results/training_history.csv](results/training_history.csv)
- [results/attack_budget_accuracy.csv](results/attack_budget_accuracy.csv)
- [results/attack_budget_accuracy.png](results/attack_budget_accuracy.png)
- [results/edge_drop_sweep.csv](results/edge_drop_sweep.csv)
- [results/edge_drop_tradeoff.png](results/edge_drop_tradeoff.png)
- [results/sparse_edge_flip_sweep.csv](results/sparse_edge_flip_sweep.csv)
- [results/sparse_flip_tradeoff.png](results/sparse_flip_tradeoff.png)
- [results/local_certificate_sweep.csv](results/local_certificate_sweep.csv)
- [results/local_certificate_radius.png](results/local_certificate_radius.png)
- [results/symmetric_certificate_sweep.csv](results/symmetric_certificate_sweep.csv)
- [results/certificate_subset_rows.csv](results/certificate_subset_rows.csv)
- [results/certified_accuracy_curve.csv](results/certified_accuracy_curve.csv)
- [results/certified_accuracy_curve.png](results/certified_accuracy_curve.png)