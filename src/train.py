import torch
import torch.nn.functional as F


def train_one_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()
    return loss.item()


# Evaluate base graph
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    metrics = {}
    for split_name, mask in {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }.items():
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        metrics[split_name] = correct / total if total > 0 else 0.0

    return metrics, pred


# Used for evaluating perturbed graph
@torch.no_grad()
def evaluate_with_edge_index(model, data, edge_index, edge_weight=None):
    model.eval()
    out = model(data.x, edge_index, edge_weight=edge_weight)
    pred = out.argmax(dim=1)

    metrics = {}
    for split_name, mask in {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }.items():
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        metrics[split_name] = correct / total if total > 0 else 0.0

    return metrics, pred


# Used for randomized smoothing training
@torch.no_grad()
def evaluate_smoothed(model, data, num_samples=1000, p_drop=0.1, batch_size=100):
    from src.smoothing import smoothed_predict

    vote_counts, smoothed_pred = smoothed_predict(
        model=model,
        data=data,
        num_samples=num_samples,
        p_drop=p_drop,
        batch_size=batch_size,
    )

    metrics = {}
    for split_name, mask in {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }.items():
        correct = (smoothed_pred[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        metrics[split_name] = correct / total if total > 0 else 0.0

    return metrics, vote_counts, smoothed_pred

@torch.no_grad()
def evaluate_smoothed_with_edge_index(model, data, edge_index, num_samples=500, p_drop=0.1, batch_size=50):
    from src.smoothing import smoothed_predict_with_edge_index

    vote_counts, smoothed_pred = smoothed_predict_with_edge_index(
        model=model,
        x=data.x,
        edge_index=edge_index,
        num_samples=num_samples,
        p_drop=p_drop,
        batch_size=batch_size,
    )

    metrics = {}
    for split_name, mask in {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }.items():
        correct = (smoothed_pred[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        metrics[split_name] = correct / total if total > 0 else 0.0

    return metrics, vote_counts, smoothed_pred