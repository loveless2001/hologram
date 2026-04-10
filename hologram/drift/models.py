from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DriftDimension:
    name: str
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftComparisonInput:
    baseline_id: str
    target_id: str
    baseline_items: List[Dict[str, Any]]
    target_items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    baseline_id: str
    target_id: str
    drift_score: float
    confidence: float
    dimensions: List[DriftDimension]
    attribution: List[str] = field(default_factory=list)
    missing_signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
