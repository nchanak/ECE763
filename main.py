import random
import numpy as np
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from src.models import GCN
from src.train import train_one_epoch, evaluate, evaluate_with_edge_index, evaluate_smoothed_with_edge_index
from src.attack import run_prbcd_attack
from src.nettack import choose_correct_test_nodes,train_deeprobust_surrogate,run_nettack_on_node
from src.train import evaluate_smoothed
from src.smoothing import certify_node_from_votes

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(
        root="data/Planetoid",
        name="Cora",
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=32,
        out_channels=dataset.num_classes,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, 201):
        loss = train_one_epoch(model, data, optimizer)
        metrics, _ = evaluate(model, data)

        if metrics["val"] > best_val:
            best_val = metrics["val"]
            best_test = metrics["test"]

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Train {metrics['train']:.4f} | "
                f"Val {metrics['val']:.4f} | "
                f"Test {metrics['test']:.4f}"
            )

    print(f"\nBest Val:  {best_val:.4f}")
    print(f"Best Test: {best_test:.4f}")

    # clean eval
    clean_metrics, _ = evaluate_with_edge_index(model, data, data.edge_index)
    print(f"\nClean test accuracy: {clean_metrics['test']:.4f}")
    
    # Smooth Eval
    print("\n=== Smoothed classifier evaluation ===")
    smoothed_metrics, vote_counts, smoothed_pred = evaluate_smoothed(
        model,
        data,
        num_samples=500,
        p_drop=0.1,
        batch_size=50,
    )

    print(
        f"Smoothed Train: {smoothed_metrics['train']:.4f} | "
        f"Smoothed Val: {smoothed_metrics['val']:.4f} | "
        f"Smoothed Test: {smoothed_metrics['test']:.4f}"
    )

    # attack eval
    for budget in [5, 10, 20, 50]:
        attacked_edge_index, flipped_edges = run_prbcd_attack(model, data, budget, device)
        attacked_metrics, _ = evaluate_with_edge_index(model, data, attacked_edge_index)

        num_flips = flipped_edges.size(1) if flipped_edges.numel() > 0 else 0
        print(
            f"Budget {budget:>3d} | "
            f"Attacked test accuracy: {attacked_metrics['test']:.4f} | "
            f"Flipped edges: {num_flips}"
        )
    
    print("\n=== Nettack targeted demo ===")

    surrogate, adj, features, labels = train_deeprobust_surrogate(data, device=device)
    candidate_nodes = choose_correct_test_nodes(model, data, k=5)

    # just take the first successful attacked node for comparison
    nettack_result = None

    for target_node in candidate_nodes:
        result = run_nettack_on_node(
            target_model=model,
            data=data,
            surrogate=surrogate,
            adj=adj,
            features=features,
            labels=labels,
            target_node=target_node,
            n_perturbations=3,
            device=device,
        )

        print(
            f"Node {result['target_node']} | "
            f"True {result['true_label']} | "
            f"Clean pred {result['clean_pred']} | "
            f"Attacked pred {result['attacked_pred']} | "
            f"Success: {result['success']}"
        )
        print("Structure perturbations:", result["perturbations"])

        if result["success"] and nettack_result is None:
            nettack_result = result

    if nettack_result is not None:
        node_idx = nettack_result["target_node"]
        attacked_edge_index = nettack_result["attacked_edge_index"]

    print("\n=== Clean vs Nettack smoothed comparison (p_drop sweep) ===")

    p_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6]

    for p in p_values:
        print(f"\n--- p_drop = {p} ---")

        clean_s_metrics, clean_vote_counts, clean_s_pred = evaluate_smoothed_with_edge_index(
            model=model,
            data=data,
            edge_index=data.edge_index,
            num_samples=500,
            p_drop=p,
            batch_size=50,
        )

        attack_s_metrics, attack_vote_counts, attack_s_pred = evaluate_smoothed_with_edge_index(
            model=model,
            data=data,
            edge_index=attacked_edge_index,
            num_samples=500,
            p_drop=p,
            batch_size=50,
        )

        clean_cert = certify_node_from_votes(clean_vote_counts, node_idx)
        attack_cert = certify_node_from_votes(attack_vote_counts, node_idx)

        print(f"Clean top class: {clean_cert['top_class']} | "
            f"pA={clean_cert['pA_hat']:.3f} | margin={clean_cert['margin']:.3f}")

        print(f"Attack top class: {attack_cert['top_class']} | "
            f"pA={attack_cert['pA_hat']:.3f} | margin={attack_cert['margin']:.3f}")

        print(f"Clean pred: {clean_s_pred[node_idx].item()} | "
            f"Attack pred: {attack_s_pred[node_idx].item()} | "
            f"True: {data.y[node_idx].item()}")

        print(f"Clean test acc: {clean_s_metrics['test']:.4f} | "
            f"Attack test acc: {attack_s_metrics['test']:.4f}")

if __name__ == "__main__":
    main()