Install the dependencies in [requirements.txt](requirements.txt) before running the scripts.

Current Project State

- Baseline GCN training on the Cora Planetoid dataset is implemented in [main.py](main.py). The current run uses 2708 nodes and 5278 undirected edges.

- Two attack paths are implemented:
	PRBCD global structural attack in [src/attack.py](src/attack.py)
	Nettack targeted attack in [src/nettack.py](src/nettack.py)

- Randomized smoothing now supports:
	edge-drop smoothing
	sparse edge-flip smoothing with separate deletion and addition probabilities
	local symmetric edge-flip smoothing for node-level certificate experiments

- Certificate utilities are implemented in [src/smoothing.py](src/smoothing.py). The current code reports node-level certified radii and certified-accuracy curves, but only counts radii as reportable when the smoothed prediction is correct.

- Result artifacts are written automatically to [results](results), including CSV summaries, PNG plots, and [results/preliminary_results.json](results/preliminary_results.json).

Latest Preliminary Results

- Clean test accuracy: 0.806

- Edge-drop smoothing baseline test accuracy: 0.805

- Tuned sparse edge-flip smoothing baseline test accuracy: 0.806 with `p_delete=0.02`, `p_add=1e-5`, and `max_additions=96`

- PRBCD global attack degrades test accuracy from 0.803 at budget 5 to 0.764 at budget 50.

- Nettack still finds successful targeted attacks on sampled nodes.

- The new local certified-accuracy subset sweep is currently weak: on a 10-node subset of correctly classified base-model test nodes, the best observed certified accuracy is 0.20 at radius 0 for `p_flip=0.01`, and 0.0 for radius at least 1 across the current sweep.

What Was Addressed

- Added a reportable certified-radius path instead of only vote margins.

- Added sparse edge-flip smoothing with separate deletion and addition probabilities.

- Added structured experiment outputs for preliminary results instead of console-only text.

- Added plots for attack accuracy, smoothing tradeoffs, certified accuracy, and local certificate sweeps.

Remaining Gaps

- The current certified-accuracy experiment is still limited to a local target-node subset and symmetric local edge-flip noise. It is not yet a strong certificate under sparsity-aware asymmetric smoothing.

- The local symmetric certificate remains pessimistic on Cora, which is consistent with the proposal motivation for sparsity-aware smoothing.

- The selected Nettack case-study node is still misclassified by the local symmetric smoothed classifier, so the reportable local radius stays at 0 in that comparison.

- More presentation-ready analysis is still possible, especially a short written summary that cites the saved artifacts directly.