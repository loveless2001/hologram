"""Prototype embedding classifier for router fallback decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .prototypes import INTENT_PROTOTYPES


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    scores: Dict[str, float]


class PrototypeIntentClassifier:
    """Embed text and compare it against pre-seeded intent centroids."""

    def __init__(
        self,
        embed_text: Callable[[str], np.ndarray],
        prototypes: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._embed_text = embed_text
        self._prototypes = prototypes or INTENT_PROTOTYPES
        self._centroids: Optional[Dict[str, np.ndarray]] = None

    def _ensure_centroids(self) -> Dict[str, np.ndarray]:
        if self._centroids is not None:
            return self._centroids

        centroids: Dict[str, np.ndarray] = {}
        for intent, examples in self._prototypes.items():
            vectors = []
            for example in examples:
                vec = np.asarray(self._embed_text(example), dtype="float32")
                vec /= (np.linalg.norm(vec) + 1e-8)
                vectors.append(vec)
            centroid = np.mean(vectors, axis=0).astype("float32")
            centroid /= (np.linalg.norm(centroid) + 1e-8)
            centroids[intent] = centroid
        self._centroids = centroids
        return centroids

    def classify(self, text: str) -> ClassificationResult:
        clean = text.strip()
        if not clean:
            return ClassificationResult(
                intent="fallback",
                confidence=0.0,
                scores={"fallback": 0.0},
            )

        query = np.asarray(self._embed_text(clean), dtype="float32")
        query /= (np.linalg.norm(query) + 1e-8)

        scores: Dict[str, float] = {}
        centroids = self._ensure_centroids()
        for intent, centroid in centroids.items():
            scores[intent] = float(np.dot(query, centroid))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0

        absolute = (top_score + 1.0) / 2.0
        margin = max(0.0, top_score - second_score)
        confidence = min(0.99, max(0.05, 0.6 * absolute + 0.8 * margin))

        return ClassificationResult(
            intent=top_intent,
            confidence=round(float(confidence), 4),
            scores=scores,
        )
