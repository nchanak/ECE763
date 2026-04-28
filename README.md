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

- Certificate utilities are implemented in [src/smoothing.py](src/smoothing.py). The current code reports node-level certified radii and certified-accuracy curves, but only counts radii as reportable when the smoothed prediction is correct. Sparse edge-flip certificates now track deletion and addition budgets separately, then report a conservative total radius. Certified-accuracy sweeps default to a lightweight 10-node, 100-sample setting; use the CLI flags below for larger final-report runs.

- Result artifacts are written automatically to [results](results), including CSV summaries, PNG plots, and [results/preliminary_results.json](results/preliminary_results.json).

- Nettack examples are treated as attack case studies. Clean certificate case studies are selected separately from correctly classified smoothed test nodes and saved to `clean_certificate_focus_sweep.csv`. Sparse certificate outputs also include total, deletion-only, and addition-only certified accuracy.

Latest Preliminary Results

- Clean test accuracy: 0.806

- Impose sparsity on edge-flip smoothing by making the probability of deletion larger than the probability of adding an edge, so deletion and addition use separate probability distributions.

Useful Commands

- Single-seed default run:
	`python main.py --seed 42 --output-dir results`

- Multi-seed run with aggregate mean/std reporting:
	`python main.py --seeds 42,43,44 --output-dir results/multiseed`

- Final-style certified-accuracy run:
	`python main.py --seed 42 --certified-samples 100 --certificate-node-count 10 --output-dir results/final_seed_42`

- Quick smoke run:
	`python main.py --seeds 1,2 --epochs 1 --attack-budgets none --nettack-targets 0 --smoothing-modes edge-drop --edge-drop-sweep 0.05 --output-dir results/smoke_cli --certified-samples 1 --certificate-node-count 1`

- Full test-mask certification, which is much slower:
	`python main.py --certified-samples 1000 --certificate-node-count none`


## Large Changes

This update turns the project from a fixed single-run experiment script into a configurable experiment runner for certified robustness experiments on GNNs.

The main experiment pipeline in main.py now supports command-line configuration. Users can specify the dataset, seed or seed list, training epochs, output directory, PRBCD attack budgets, smoothing modes, edge-drop sweep values, sparse edge-flip sweep values, certificate sample counts, certificate node counts, and Nettack target count. Multi-seed runs now write per-seed artifact directories and aggregate results into multi_seed_summary.csv and multi_seed_summary.json.

The certification logic was extended to support sparse asymmetric edge-flip certificates. In addition to the previous symmetric local edge-flip certificate, the code now tracks deletion and addition perturbation budgets separately. Sparse certificate outputs include conservative total radius, deletion-only radius, and addition-only radius. This is important because addition robustness is much harder under very small p_add, while deletion-only certificates show meaningful nonzero robustness.

The sparse edge-flip sampler was also corrected. Previously, global sparse additions could be sampled from the graph after deletions, meaning an edge deleted earlier in the same noisy sample could be re-added. Additions are now sampled from original non-edges, matching the intended asymmetric smoothing distribution.

The experiment reporting was improved. Certified accuracy summaries now include Wilson 95% confidence intervals, and sparse asymmetric reports include total, deletion-only, and addition-only certified accuracy curves. CSV writing now handles rows with heterogeneous fields, which is needed for multi-seed and grouped summary outputs.

The target-node analysis was clarified. Nettack-selected nodes are now treated as attack case studies, not certificate case studies. A separate clean certificate focus node is selected from smoothed-correct test nodes and saved to clean_certificate_focus_sweep.csv, making the certification case study more representative and less self-defeating.

The README was updated with the new project state, corrected wording around sparse smoothing, and added practical commands for default runs, multi-seed runs, quick smoke tests, final-style certificate runs, and full test-mask certification.
