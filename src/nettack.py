import numpy as np
import scipy.sparse as sp
import torch

from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN as DRGCN

# Targeted attack, flips prediction with single edge flip
def pyg_to_scipy_adj(data) -> sp.csr_matrix:
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    vals = np.ones(len(row), dtype=np.float32)
    n = data.num_nodes

    adj = sp.csr_matrix((vals, (row, col)), shape=(n, n))
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def pyg_features_to_scipy(data) -> sp.csr_matrix:
    return sp.csr_matrix(data.x.cpu().numpy())


def scipy_adj_to_edge_index(adj: sp.csr_matrix, device):
    adj = adj.tocoo()
    row = torch.from_numpy(adj.row).long()
    col = torch.from_numpy(adj.col).long()
    return torch.stack([row, col], dim=0).to(device)


@torch.no_grad()
def predict_node(model, data, edge_index, node_idx):
    model.eval()
    out = model(data.x, edge_index)
    pred = out.argmax(dim=1)
    return pred[node_idx].item(), out[node_idx].cpu()


def choose_correct_test_nodes(model, data, k=10):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    correct = ((pred == data.y) & data.test_mask).nonzero(as_tuple=False).view(-1)
    correct = correct.cpu().tolist()
    return correct[:k]


def train_deeprobust_surrogate(data, device="cpu"):
    adj = pyg_to_scipy_adj(data)
    features = pyg_features_to_scipy(data)
    labels = data.y.cpu().numpy()

    idx_train = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    idx_val = data.val_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

    surrogate = DRGCN(
        nfeat=features.shape[1],
        nclass=int(labels.max()) + 1,
        nhid=16,
        dropout=0.0,
        with_relu=False,
        with_bias=False,
        device=device,
    ).to(device)

    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    return surrogate, adj, features, labels


def run_nettack_on_node(
    target_model,
    data,
    surrogate,
    adj,
    features,
    labels,
    target_node,
    n_perturbations=3,
    device="cpu",
):
    attack_model = Nettack(
        surrogate,
        nnodes=adj.shape[0],
        attack_structure=True,
        attack_features=False,
        device=device,
    ).to(device)

    attack_model.attack(
        features,
        adj,
        labels,
        target_node,
        n_perturbations,
        direct=True,
        verbose=False,
    )

    modified_adj = attack_model.modified_adj
    attacked_edge_index = scipy_adj_to_edge_index(modified_adj, device=device)

    clean_pred, _ = predict_node(target_model, data, data.edge_index, target_node)
    attacked_pred, _ = predict_node(target_model, data, attacked_edge_index, target_node)

    return {
        "target_node": int(target_node),
        "true_label": int(data.y[target_node].item()),
        "clean_pred": int(clean_pred),
        "attacked_pred": int(attacked_pred),
        "success": int(attacked_pred) != int(data.y[target_node].item()),
        "perturbations": attack_model.structure_perturbations,
        "attacked_edge_index": attacked_edge_index,
    }