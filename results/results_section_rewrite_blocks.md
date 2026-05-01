# Results Section Rewrite Blocks

Paste-ready replacement prose for the current Results screenshot blocks. The blocks are ordered to match the current report flow so you can copy only the pieces you want. Blocks 7 through 11 are best treated as appendix or ablation material rather than the main headline Results section.

## Block 1 - After the Cora training-variant table

The training-variant comparison shows that this project is not only evaluating a single clean baseline under attack, but also testing whether robustness-aligned training changes downstream behavior. On Cora, the clean and matched sparse-noisy models retain the same clean test accuracy of 0.817, while the warm-start symmetric fine-tuning model trades a small amount of clean accuracy for a training procedure that is more aligned with the later certification setting. The sparse-noisy variant improves validation accuracy, but that gain alone does not make it the strongest final defense.

## Block 2 - Transition into the attack results

These baseline results motivate the remainder of the section. Before turning to certification, we first verify that both global and targeted structural attacks are effective against the underlying GNNs, and then test whether purification can recover useful performance after those attacks.

## Block 3 - PRBCD global-attack figure paragraph

The PRBCD results confirm that global structural perturbations can reduce test accuracy even when the clean model starts near 81.7 percent. As the perturbation budget increases, the attacked accuracy falls overall and reaches 79.1 percent at budget 50. Although the intermediate points are not perfectly monotone, the high-budget regime clearly shows that global edge flips degrade performance and provide a meaningful stress test for downstream defenses.

## Block 4 - Nettack table paragraph

The targeted Nettack results show that local structural attacks are also effective. On the selected Cora targets, all five attacks successfully changed the model prediction away from the correct class. This matters because a defense that survives only average-case global corruption but fails on targeted manipulations would still be brittle in adversarial settings.

## Block 5 - Jaccard tradeoff figure paragraph

Jaccard-based purification provides a useful but limited recovery mechanism after global attack. Small thresholds improve the attacked graph relative to the unpurified attacked baseline, increasing attacked accuracy from 79.1 percent to 79.6 percent at thresholds 0.01 and 0.02, while reducing clean-graph accuracy from 81.7 percent to 80.5 percent. At the larger threshold 0.05, both clean and attacked performance decline, showing that overly aggressive pruning begins to remove too many informative edges.

## Block 6 - Nettack target-node Jaccard repair table paragraph

The target-node repair table shows the same tradeoff at the level of attacked Nettack nodes. Without purification, none of the five targeted predictions are recovered. Thresholds 0.01 and 0.02 recover three of the five targets, while threshold 0.05 recovers four of five, but that stronger recovery comes with much lower target-edge retention. In other words, Jaccard pruning can repair targeted attacks, but the recovery is purchased by aggressively rewriting the local neighborhood around the attacked node.

## Block 7 - Transition into the certification diagnostics

The next figures should be read as certificate diagnostics rather than the final headline results. Their purpose is to explain why some smoothing and certificate families were retained only as ablations, and why the later report focuses on the stronger symmetric and top-2 purified configurations.

## Block 8 - Radius-1 certificate-type figure paragraph

At radius 1, the deletion-only asymmetric certificates are substantially stronger than the symmetric add/delete certificates for both the clean and robust models. This is consistent with the attack setting: targeted structural attacks are sparse and often easier to align with deletion-dominated certificates than with balanced add/delete perturbations. At the same time, the absence of useful addition-side certificates highlights a practical limitation of the asymmetric formulation under the current computational budget.

## Block 9 - Clean GCN certified-curves paragraph

For the clean GCN, certified accuracy remains limited under symmetric add/delete smoothing, but the deletion-only asymmetric view can still certify a meaningful fraction of nodes at small radii. These curves show that certification is possible even without robust training, but the resulting guarantees are narrow and decay quickly as the requested perturbation size increases.

## Block 10 - Sparse-noisy trained GCN certified-curves paragraph

The sparse-noisy trained GCN shows a similar qualitative pattern, but with better stability in the lower-radius regime. Robust training helps preserve certified accuracy under the deletion-only asymmetric view and supports stronger low-radius behavior, yet the curves still fall off quickly as the requested perturbation size grows. This is why the final report emphasizes positive-certified fraction and best fixed configurations rather than claiming broad large-radius robustness.

## Block 11 - High-addition ablation paragraph

High-addition smoothing regimes were tested as an ablation rather than adopted as the mainline defense family. In isolated diagnostics, increasing the addition probability can increase the nominal maximum radius, but it can also sharply reduce correctness on the evaluated nodes. For that reason, the final validated results treat high-addition settings as exploratory evidence and rely instead on the stronger deployable symmetric and top-2 configurations.

## Block 12 - Bridge from old Cora diagnostics into the new final results

Taken together, the Cora-only diagnostics establish three points: structural attacks are effective, purification can partially recover attacked performance, and certification behavior depends strongly on how the smoothing distribution is chosen. The remainder of the results section therefore shifts from single-dataset diagnostics to purified-certificate comparisons on Cora and the final cross-dataset winner-only benchmark.

## New Block A - Intro to the Cora purified-certificate comparison

The purified-certificate comparison on Cora shows that the strongest fixed configurations come from the symmetric-0.3 family and its top-2 or cosine variants. The leading clean, warm-start, and purification-aware variants all reach roughly 29.5 percent positive certification and 43.4 percent fixed-config correctness, while oracle selection increases correctness further without materially increasing the certification ceiling. This makes Cora the clearest case study for how a deployable fixed defense compares to a best-of-candidates upper bound.

## New Block B - Intro to the cross-dataset winner-only benchmark

To test whether these trends generalize beyond a single citation graph, we ran a winner-only benchmark across Cora, CiteSeer, and PubMed, using both GCN and GraphSAGE under standard and adaptive-purified attacks. This broader view shows that GraphSAGE dominates many of the strongest configurations, PubMed produces the highest overall certification rates, and adaptive-purified attacks usually reduce certification relative to the corresponding standard setting.

## New Block C - Selector-versus-oracle paragraph

The selector-versus-oracle comparison shows that there is very little untapped certification headroom inside the current candidate family. In nearly every completed combination, the selector matches the oracle positive-certified fraction exactly or leaves only a negligible gap. This means the main challenge is not finding a dramatically better per-node candidate after the fact, but improving the underlying candidate family itself.

## New Block D - Coverage and limitations paragraph

The coverage results should be read alongside the headline certification numbers. Although the final winner-only suite is complete across datasets, many adaptive combinations did not reach the full 20-target evaluation goal, especially on CiteSeer and some GraphSAGE settings. These target-pool gaps do not invalidate the completed results, but they do limit how strongly some adaptive comparisons should be interpreted and should be treated as an explicit limitation of the current benchmark.

## New Block E - Closing paragraph for the full Results section

Overall, the final results support a consistent conclusion across the validated artifacts. Purification and randomized smoothing can recover meaningful certified performance after structural attack, but the strongest deployable results come from a relatively small family of symmetric and top-2 configurations rather than from the more attack-aligned low-noise asymmetric branch. The cross-dataset benchmark further shows that these trends are not confined to Cora alone, with the strongest certification rates appearing on PubMed and the strongest architecture-level results often coming from GraphSAGE.