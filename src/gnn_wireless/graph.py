from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WirelessGraph:
    features: np.ndarray
    adjacency: np.ndarray
    labels: np.ndarray
    positions: np.ndarray


def greedy_schedule(adjacency: np.ndarray, priorities: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(priorities), dtype=np.float32)
    for node in np.argsort(priorities)[::-1]:
        neighbors = np.flatnonzero(adjacency[node] > 0)
        if not np.any(selected[neighbors] > 0):
            selected[node] = 1.0
    return selected


def generate_graph(
    n_nodes: int = 32,
    *,
    interference_radius: float = 0.25,
    seed: int = 7,
) -> WirelessGraph:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, size=(n_nodes, 2))
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    adjacency = ((distance < interference_radius) & (distance > 0)).astype(np.float32)
    demand = rng.uniform(0.2, 1.0, n_nodes)
    base_station = np.array([0.5, 0.5])
    bs_distance = np.linalg.norm(positions - base_station, axis=1) + 0.05
    snr_proxy = np.clip(1.0 - bs_distance / np.sqrt(0.5), 0.0, 1.0)
    degree = adjacency.sum(axis=1)
    degree_norm = degree / max(float(degree.max()), 1.0)
    features = np.column_stack([demand, snr_proxy, degree_norm]).astype(np.float32)
    priorities = demand * (0.5 + snr_proxy) / (1.0 + degree)
    labels = greedy_schedule(adjacency, priorities)
    return WirelessGraph(features, adjacency, labels, positions.astype(np.float32))
