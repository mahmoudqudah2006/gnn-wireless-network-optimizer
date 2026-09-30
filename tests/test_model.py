import torch

from gnn_wireless.graph import generate_graph
from gnn_wireless.model import GCN


def test_model_output_shape() -> None:
    graph = generate_graph(12, seed=3)
    model = GCN()
    logits = model(torch.from_numpy(graph.features), torch.from_numpy(graph.adjacency))
    assert logits.shape == (12,)
