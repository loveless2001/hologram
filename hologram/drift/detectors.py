from __future__ import annotations

from typing import Callable, List
import numpy as np

from .models import DriftDimension
from ..kg.models import BatchKGSnapshot


def _mean_centroid(vectors: List[np.ndarray]) -> np.ndarray:
    mat = np.stack(vectors).astype("float32")
    centroid = mat.mean(axis=0)
    norm = np.linalg.norm(centroid) + 1e-8
    return centroid / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (an * bn))


def embedding_centroid_drift(
    baseline_texts: List[str],
    target_texts: List[str],
    embed_func: Callable[[str], np.ndarray],
) -> DriftDimension:
    if not baseline_texts or not target_texts:
        return DriftDimension(
            name="embedding_centroid",
            score=0.0,
            confidence=0.0,
            details={"reason": "missing_texts"},
        )

    base_vecs = [embed_func(t) for t in baseline_texts]
    targ_vecs = [embed_func(t) for t in target_texts]

    base_c = _mean_centroid(base_vecs)
    targ_c = _mean_centroid(targ_vecs)

    # Convert cosine similarity to drift distance in [0, 1].
    score = max(0.0, min(1.0, (1.0 - _cosine(base_c, targ_c)) / 2.0))
    support = min(len(base_vecs), len(targ_vecs))
    confidence = max(0.0, min(1.0, support / 20.0))

    return DriftDimension(
        name="embedding_centroid",
        score=float(score),
        confidence=float(confidence),
        details={"baseline_count": len(base_vecs), "target_count": len(targ_vecs)},
    )


def kg_structure_drift(baseline: BatchKGSnapshot, target: BatchKGSnapshot) -> DriftDimension:
    base_nodes = {n.label for n in baseline.nodes}
    targ_nodes = {n.label for n in target.nodes}

    base_edges = {(e.source, e.target, e.relation) for e in baseline.edges}
    targ_edges = {(e.source, e.target, e.relation) for e in target.edges}

    node_union = base_nodes | targ_nodes
    edge_union = base_edges | targ_edges

    node_churn = 0.0 if not node_union else 1.0 - (len(base_nodes & targ_nodes) / len(node_union))
    edge_churn = 0.0 if not edge_union else 1.0 - (len(base_edges & targ_edges) / len(edge_union))

    score = 0.6 * node_churn + 0.4 * edge_churn
    support = min(len(base_nodes), len(targ_nodes))
    confidence = max(0.0, min(1.0, support / 25.0))

    return DriftDimension(
        name="kg_structure",
        score=float(score),
        confidence=float(confidence),
        details={
            "node_churn": round(float(node_churn), 6),
            "edge_churn": round(float(edge_churn), 6),
            "baseline_nodes": len(base_nodes),
            "target_nodes": len(targ_nodes),
            "baseline_edges": len(base_edges),
            "target_edges": len(targ_edges),
        },
    )
