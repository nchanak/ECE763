Make sure to install the requirements.txt for the scripts to function

Done:

- Implemented GNN with torch sample data, about 10,000 edges and 5000 nodes

- Performed two attacks on GNN, one global attack which attacks 10-50 edges and shows some drops in accuracy,
the other nettack shows how flipping a single edge can change a prediction

- Implemented some random smoothing which while it can't avoid nettack's flipped prediction, it currently
makes the prediction less confident, perhaps different tuning could change this.

Need:

- Right now we get some basic stats on the randomized smoothing, we don't have the certified radius metric

- Our randomized smoothing just does edge-drop smoothing, probably implement edge-flip smoothing

- Could use cleaner metrics in general to present in the phase 2 submission rather than just python text,
perhaps some graphs or something

- Impose smarsity on edge-flip smoothing by making probability of deletion larger than probability of adding edge,
so the deletion and addition use separate probability distributions
