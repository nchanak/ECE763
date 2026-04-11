import torch
from torch_geometric.contrib.nn import PRBCDAttack

# Global based attack, general decrease in model accuracy
def run_prbcd_attack(model, data, budget, device):
    model.eval()

    attack = PRBCDAttack(
        model=model,
        block_size=20000,
        epochs=100,
        epochs_resampling=20,
        lr=200.0,
        is_undirected=True,
        log=True,
    ).to(device)

    attacked_edge_index, flipped_edges = attack.attack(
        x=data.x,
        edge_index=data.edge_index,
        labels=data.y,
        budget=budget,
        idx_attack=data.test_mask,
    )

    return attacked_edge_index, flipped_edges