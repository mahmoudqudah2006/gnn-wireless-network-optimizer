import numpy as np

from gnn_wireless.graph import generate_graph
from gnn_wireless.scheduler import conflict_count, project_conflict_free


def test_graph_adjacency_is_symmetric() -> None:
    graph = generate_graph(20, seed=4)
    assert np.array_equal(graph.adjacency, graph.adjacency.T)
    assert np.all(np.diag(graph.adjacency) == 0)


def test_projection_is_conflict_free() -> None:
    graph = generate_graph(30, seed=9)
    scores = np.linspace(0, 1, 30)
    selection = project_conflict_free(graph.adjacency, scores)
    assert conflict_count(graph.adjacency, selection) == 0
