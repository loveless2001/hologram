from __future__ import annotations

from typing import Any, Dict, List, Optional

from .detectors import embedding_centroid_drift, kg_structure_drift
from .models import DriftComparisonInput, DriftReport
from ..kg.builder import build_batch_kg_snapshot


def _collect_texts(items: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for item in items:
        value = item.get("payload", item.get("text", ""))
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def compare_batches(
    comparison: DriftComparisonInput,
    *,
    embed_func,
    baseline_snapshot=None,
    target_snapshot=None,
) -> DriftReport:
    baseline_texts = _collect_texts(comparison.baseline_items)
    target_texts = _collect_texts(comparison.target_items)

    dim_embedding = embedding_centroid_drift(baseline_texts, target_texts, embed_func)

    if baseline_snapshot is None:
        baseline_snapshot = build_batch_kg_snapshot(comparison.baseline_id, comparison.baseline_items)
    if target_snapshot is None:
        target_snapshot = build_batch_kg_snapshot(comparison.target_id, comparison.target_items)

    dim_kg = kg_structure_drift(baseline_snapshot, target_snapshot)

    dimensions = [dim_embedding, dim_kg]

    weights = {
        "embedding_centroid": 0.55,
        "kg_structure": 0.45,
    }

    weighted_score = 0.0
    weighted_conf = 0.0
    weight_sum = 0.0
    for dim in dimensions:
        w = weights.get(dim.name, 0.0)
        weighted_score += w * dim.score
        weighted_conf += w * dim.confidence
        weight_sum += w

    drift_score = 0.0 if weight_sum == 0 else weighted_score / weight_sum
    confidence = 0.0 if weight_sum == 0 else weighted_conf / weight_sum

    attribution = sorted(dimensions, key=lambda d: d.score, reverse=True)
    attribution_lines = [
        f"{d.name}: score={d.score:.3f}, confidence={d.confidence:.3f}"
        for d in attribution
    ]

    missing_signals = []
    if not baseline_texts or not target_texts:
        missing_signals.append("text_payload")

    metadata = {
        "baseline_items": len(comparison.baseline_items),
        "target_items": len(comparison.target_items),
        "baseline_nodes": len(baseline_snapshot.nodes),
        "target_nodes": len(target_snapshot.nodes),
    }

    return DriftReport(
        baseline_id=comparison.baseline_id,
        target_id=comparison.target_id,
        drift_score=float(drift_score),
        confidence=float(confidence),
        dimensions=dimensions,
        attribution=attribution_lines,
        missing_signals=missing_signals,
        metadata=metadata,
    )
