# GNN Wireless Network Optimizer

A research-oriented **Graph Neural Network (GNN)** baseline for conflict-aware wireless scheduling on synthetic interference graphs.

Nodes represent wireless links/users. Edges represent interference conflicts. The learning task is to imitate a transparent greedy scheduler and predict node priorities that can be converted into a conflict-free schedule.

## Why graphs?

Wireless resource-allocation problems are naturally relational: whether one user can be scheduled depends on neighboring interferers, not only on its own features. A graph representation makes those dependencies explicit.

## Graph construction

Each synthetic graph contains random node positions. Two nodes are connected when their Euclidean distance is below an interference radius.

Node features are:

1. normalized traffic demand
2. normalized SNR proxy
3. normalized graph degree

A deterministic greedy heuristic generates supervision labels by prioritizing high demand with a degree penalty.

## GCN model

To keep the implementation inspectable, this repository implements the normalized GCN operation directly in PyTorch:

[
H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)}\right),
]

where (\hat{A}=A+I).

This mirrors the standard graph-convolution formulation and avoids requiring PyTorch Geometric for the first version. A future adapter can use `torch_geometric.nn.GCNConv` for larger experiments.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

gnn-wireless train --graphs 300 --epochs 40 --seed 7 --output results/training.json
```

## Evaluation

The training script reports:

- node-level classification accuracy
- predicted schedule size
- conflict violations after raw thresholding
- conflict-free schedule size after greedy projection

The conflict-free projection is important: a classifier score alone is not a valid wireless schedule.

## Scope

This is a synthetic research baseline, not a production RRM optimizer. It provides a clean path toward:

- channel allocation
- power control
- link scheduling
- interference coordination
- heterogeneous graph features
- temporal GNNs
- graph reinforcement learning

## License

MIT

---

**Mahmoud Alqudah** · Graph Neural Networks · Wireless Optimization · AI for Communications
