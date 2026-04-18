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

- Impose smarsity on edge-flip smoothing by making probability of deletion larger than probability of adding edge,
so the deletion and addition use separate probability distributions
