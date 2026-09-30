from __future__ import annotations

import numpy as np

from .graph import greedy_schedule


def project_conflict_free(adjacency: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return greedy_schedule(adjacency, np.asarray(scores, dtype=float))


def conflict_count(adjacency: np.ndarray, selection: np.ndarray) -> int:
    chosen = np.flatnonzero(np.asarray(selection) > 0)
    if len(chosen) < 2:
        return 0
    subgraph = adjacency[np.ix_(chosen, chosen)]
    return int(subgraph.sum() // 2)
