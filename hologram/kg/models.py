from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any


@dataclass
class KGNode:
    node_id: str
    label: str
    node_type: str = "concept"
    weight: float = 1.0
    evidence_count: int = 1
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str = "co_occurs"
    weight: float = 1.0
    evidence_count: int = 1
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchKGSnapshot:
    batch_id: str
    timestamp: str
    nodes: List[KGNode]
    edges: List[KGEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
