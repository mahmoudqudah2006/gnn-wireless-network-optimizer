from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .graph import generate_graph
from .model import GCN
from .scheduler import conflict_count, project_conflict_free


def train_model(graphs: int = 200, epochs: int = 30, seed: int = 7) -> tuple[GCN, dict[str, Any]]:
    torch.manual_seed(seed)
    train_graphs = [generate_graph(seed=seed + i) for i in range(graphs)]
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    positives = sum(float(graph.labels.sum()) for graph in train_graphs)
    total_nodes = sum(len(graph.labels) for graph in train_graphs)
    negatives = total_nodes - positives
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for graph in train_graphs:
            x = torch.from_numpy(graph.features)
            adjacency = torch.from_numpy(graph.adjacency)
            labels = torch.from_numpy(graph.labels)
            optimizer.zero_grad()
            logits = model(x, adjacency)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss / len(train_graphs))

    test_graph = generate_graph(seed=seed + 100_000)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(test_graph.features), torch.from_numpy(test_graph.adjacency)
        )
        probabilities = torch.sigmoid(logits).numpy()
    raw = (probabilities >= 0.5).astype(np.float32)
    projected = project_conflict_free(test_graph.adjacency, probabilities)
    accuracy = float(np.mean(raw == test_graph.labels))
    true_positive = int(np.sum((raw == 1) & (test_graph.labels == 1)))
    false_positive = int(np.sum((raw == 1) & (test_graph.labels == 0)))
    false_negative = int(np.sum((raw == 0) & (test_graph.labels == 1)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return model, {
        "graphs": graphs,
        "epochs": epochs,
        "final_loss": losses[-1],
        "test_node_accuracy": accuracy,
        "test_f1": float(f1),
        "raw_conflicts": conflict_count(test_graph.adjacency, raw),
        "raw_schedule_size": int(raw.sum()),
        "projected_conflicts": conflict_count(test_graph.adjacency, projected),
        "projected_schedule_size": int(projected.sum()),
    }
