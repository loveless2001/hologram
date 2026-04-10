from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional

from .models import BatchKGSnapshot, KGEdge, KGNode
from ..text_utils import extract_concepts


def _canonicalize(term: str) -> str:
    return " ".join(term.lower().strip().split())


def build_batch_kg_snapshot(
    batch_id: str,
    items: List[Dict[str, Any]],
    *,
    timestamp: Optional[str] = None,
    min_edge_weight: int = 1,
) -> BatchKGSnapshot:
    """
    Build a lightweight semantic graph for a batch of query/response items.

    Required per item: payload (or text).
    Optional per item: id, item_type, domain, model, version, metadata.
    """
    now = timestamp or datetime.now(timezone.utc).isoformat()

    node_counts: Counter[str] = Counter()
    node_provenance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"item_ids": []})
    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_provenance: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {"item_ids": []})

    for idx, item in enumerate(items):
        raw_text = str(item.get("payload", item.get("text", ""))).strip()
        if not raw_text:
            continue

        item_id = str(item.get("id", f"item:{idx + 1}"))
        concepts = [_canonicalize(c) for c in extract_concepts(raw_text)]
        concepts = [c for c in concepts if c]

        # De-duplicate terms at item scope before building co-occurrence edges.
        uniq = sorted(set(concepts))

        for concept in uniq:
            node_counts[concept] += 1
            node_provenance[concept]["item_ids"].append(item_id)

        for a, b in combinations(uniq, 2):
            key = (a, b) if a <= b else (b, a)
            edge_counts[key] += 1
            edge_provenance[key]["item_ids"].append(item_id)

    nodes = [
        KGNode(
            node_id=f"concept:{abs(hash(label)) % 10**10}",
            label=label,
            weight=float(count),
            evidence_count=int(count),
            provenance=node_provenance[label],
        )
        for label, count in node_counts.items()
    ]

    edges = [
        KGEdge(
            source=a,
            target=b,
            relation="co_occurs",
            weight=float(count),
            evidence_count=int(count),
            provenance=edge_provenance[(a, b)],
        )
        for (a, b), count in edge_counts.items()
        if count >= min_edge_weight
    ]

    metadata = {
        "item_count": len(items),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    return BatchKGSnapshot(batch_id=batch_id, timestamp=now, nodes=nodes, edges=edges, metadata=metadata)
