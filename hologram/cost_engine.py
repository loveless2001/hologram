# hologram/cost_engine.py
"""
Cost Engine: A diagnostic meta-layer that measures cognitive effort metrics.

The Cost Engine observes the Gravity Field and computes:
- Resistance: How "out of place" a concept is in its neighborhood
- Entropy: How many competing interpretations exist (ambiguity)
- Drift Cost: Instability of concept representation over time

It does NOT mutate vectors. All suggestions are advisory signals.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from .gravity import Gravity, Concept, cosine


@dataclass
class CostEngineConfig:
    """Configuration for Cost Engine thresholds and behavior."""
    
    # Suggestion thresholds
    entropy_split_threshold: float = 0.5
    resistance_split_threshold: float = 0.4
    resistance_fuse_threshold: float = 0.2
    drift_stabilize_threshold: float = 0.3
    
    # Neighbor calculation
    neighbor_k: int = 10  # Number of neighbors to consider
    min_neighbor_sim: float = 0.1  # Minimum similarity to count as neighbor
    
    @classmethod
    def creative(cls) -> "CostEngineConfig":
        """Loose thresholds → fewer splits, more fusion."""
        return cls(
            entropy_split_threshold=0.7,
            resistance_split_threshold=0.6,
            resistance_fuse_threshold=0.3,
            drift_stabilize_threshold=0.4
        )
    
    @classmethod
    def conservative(cls) -> "CostEngineConfig":
        """Tight thresholds → more stable, fewer changes."""
        return cls(
            entropy_split_threshold=0.3,
            resistance_split_threshold=0.3,
            resistance_fuse_threshold=0.1,
            drift_stabilize_threshold=0.2
        )
    
    @classmethod
    def analytical(cls) -> "CostEngineConfig":
        """Balanced defaults for diagnostic use."""
        return cls()  # Uses default values


@dataclass
class CostReport:
    """Result of a cost evaluation."""
    
    resistance: float       # 1 - avg(similarity_to_neighbors)
    entropy: float          # 1 - cluster_coherence
    drift_cost: float       # norm(current_vec - previous_vec)
    total_cost: float       # resistance * (1 + entropy) * (1 + drift_cost)
    suggestion: str         # "split" | "fuse" | "stabilize" | "no-action"
    
    # Optional metadata
    neighbor_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "resistance": round(self.resistance, 4),
            "entropy": round(self.entropy, 4),
            "drift_cost": round(self.drift_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "suggestion": self.suggestion,
            "neighbor_count": self.neighbor_count,
            "details": self.details
        }


class CostEngine:
    """
    Diagnostic layer that computes cognitive effort metrics.
    
    Usage:
        engine = CostEngine(CostEngineConfig.analytical())
        report = engine.evaluate("concept_name", gravity_instance)
        print(report.total_cost, report.suggestion)
    """
    
    def __init__(self, config: Optional[CostEngineConfig] = None):
        self.config = config or CostEngineConfig.analytical()
    
    def _get_neighbors(self, vec: np.ndarray, gravity: Gravity, 
                       exclude_name: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get k nearest neighbors with their similarities."""
        if not gravity.concepts:
            return []
        
        neighbors = []
        for name, concept in gravity.concepts.items():
            if name == exclude_name:
                continue
            if concept.canonical_id is not None:
                continue  # Skip aliases
            
            sim = cosine(vec, concept.vec)
            if sim >= self.config.min_neighbor_sim:
                neighbors.append((name, sim))
        
        # Sort by similarity descending, take top k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:self.config.neighbor_k]
    
    def _compute_resistance(self, vec: np.ndarray, neighbors: List[Tuple[str, float]]) -> float:
        """
        Resistance = 1 - avg(similarity_to_neighbors)
        
        High resistance → concept is "out of place"
        """
        if not neighbors:
            return 1.0  # No neighbors = maximum resistance
        
        avg_sim = sum(sim for _, sim in neighbors) / len(neighbors)
        return 1.0 - avg_sim
    
    def _compute_entropy(self, neighbors: List[Tuple[str, float]], 
                         gravity: Gravity) -> float:
        """
        Entropy = 1 - cluster_coherence
        
        Cluster coherence = average pairwise similarity among neighbors.
        High entropy → neighbors are diverse/contradictory.
        """
        if len(neighbors) < 2:
            return 0.0  # Not enough neighbors to measure coherence
        
        # Compute pairwise similarities among neighbors
        neighbor_vecs = []
        for name, _ in neighbors:
            if name in gravity.concepts:
                neighbor_vecs.append(gravity.concepts[name].vec)
        
        if len(neighbor_vecs) < 2:
            return 0.0
        
        total_sim = 0.0
        count = 0
        for i in range(len(neighbor_vecs)):
            for j in range(i + 1, len(neighbor_vecs)):
                total_sim += cosine(neighbor_vecs[i], neighbor_vecs[j])
                count += 1
        
        if count == 0:
            return 0.0
        
        coherence = total_sim / count
        return 1.0 - coherence
    
    def _compute_drift_cost(self, concept: Concept) -> float:
        """
        Drift Cost = norm(current_vec - previous_vec)
        
        High drift → concept is unstable.
        """
        if concept.previous_vec is None:
            return 0.0  # No previous state = no drift
        
        diff = concept.vec - concept.previous_vec
        return float(np.linalg.norm(diff))
    
    def _compute_suggestion(self, resistance: float, entropy: float, 
                            drift_cost: float, has_overlap: bool = False) -> str:
        """Determine advisory suggestion based on metrics."""
        cfg = self.config
        
        if entropy > cfg.entropy_split_threshold and resistance > cfg.resistance_split_threshold:
            return "split"
        elif resistance < cfg.resistance_fuse_threshold and has_overlap:
            return "fuse"
        elif drift_cost > cfg.drift_stabilize_threshold:
            return "stabilize"
        else:
            return "no-action"
    
    def evaluate(self, name: str, gravity: Gravity) -> CostReport:
        """
        Evaluate cognitive cost for a specific concept.
        
        Args:
            name: Concept name to evaluate
            gravity: Gravity instance containing the concept
            
        Returns:
            CostReport with metrics and suggestion
        """
        if name not in gravity.concepts:
            return CostReport(
                resistance=1.0,
                entropy=0.0,
                drift_cost=0.0,
                total_cost=1.0,
                suggestion="no-action",
                details={"error": "concept_not_found"}
            )
        
        concept = gravity.concepts[name]
        
        # Get neighbors
        neighbors = self._get_neighbors(concept.vec, gravity, exclude_name=name)
        
        # Compute primitives
        resistance = self._compute_resistance(concept.vec, neighbors)
        entropy = self._compute_entropy(neighbors, gravity)
        drift_cost = self._compute_drift_cost(concept)
        
        # Total cost formula: resistance * (1 + entropy) * (1 + drift_cost)
        total_cost = resistance * (1.0 + entropy) * (1.0 + drift_cost)
        
        # Check for overlap (any neighbor with high similarity)
        has_overlap = any(sim > 0.8 for _, sim in neighbors)
        
        # Determine suggestion
        suggestion = self._compute_suggestion(resistance, entropy, drift_cost, has_overlap)
        
        return CostReport(
            resistance=resistance,
            entropy=entropy,
            drift_cost=drift_cost,
            total_cost=total_cost,
            suggestion=suggestion,
            neighbor_count=len(neighbors),
            details={
                "top_neighbors": [(n, round(s, 3)) for n, s in neighbors[:3]]
            }
        )
    
    def evaluate_query(self, query_vec: np.ndarray, gravity: Gravity) -> CostReport:
        """
        Evaluate cognitive cost for a query vector (not yet a concept).
        
        Useful for predicting cost before adding to the field.
        """
        # Normalize query
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Get neighbors
        neighbors = self._get_neighbors(query_vec, gravity)
        
        # Compute primitives (no drift for query)
        resistance = self._compute_resistance(query_vec, neighbors)
        entropy = self._compute_entropy(neighbors, gravity)
        drift_cost = 0.0  # No history for query
        
        total_cost = resistance * (1.0 + entropy)
        
        has_overlap = any(sim > 0.8 for _, sim in neighbors)
        suggestion = self._compute_suggestion(resistance, entropy, drift_cost, has_overlap)
        
        return CostReport(
            resistance=resistance,
            entropy=entropy,
            drift_cost=drift_cost,
            total_cost=total_cost,
            suggestion=suggestion,
            neighbor_count=len(neighbors),
            details={
                "type": "query",
                "top_neighbors": [(n, round(s, 3)) for n, s in neighbors[:3]]
            }
        )
    
    def evaluate_glyph(self, glyph_name: str, gravity: Gravity) -> CostReport:
        """
        Evaluate cognitive cost for a glyph anchor.
        
        Glyphs are identified by the "glyph:" prefix.
        """
        full_name = f"glyph:{glyph_name}" if not glyph_name.startswith("glyph:") else glyph_name
        report = self.evaluate(full_name, gravity)
        report.details["type"] = "glyph"
        return report
    
    def evaluate_field(self, gravity: Gravity) -> Dict[str, CostReport]:
        """
        Evaluate all concepts in the field.
        
        Returns a dictionary mapping concept names to their cost reports.
        """
        reports = {}
        for name, concept in gravity.concepts.items():
            if concept.canonical_id is None:  # Skip aliases
                reports[name] = self.evaluate(name, gravity)
        return reports
    
    def field_summary(self, gravity: Gravity) -> Dict[str, Any]:
        """
        Get aggregate statistics for the entire field.
        """
        reports = self.evaluate_field(gravity)
        
        if not reports:
            return {
                "concept_count": 0,
                "avg_resistance": 0.0,
                "avg_entropy": 0.0,
                "avg_drift_cost": 0.0,
                "avg_total_cost": 0.0,
                "suggestions": {}
            }
        
        avg_resistance = sum(r.resistance for r in reports.values()) / len(reports)
        avg_entropy = sum(r.entropy for r in reports.values()) / len(reports)
        avg_drift = sum(r.drift_cost for r in reports.values()) / len(reports)
        avg_total = sum(r.total_cost for r in reports.values()) / len(reports)
        
        # Count suggestions
        suggestions = {}
        for r in reports.values():
            suggestions[r.suggestion] = suggestions.get(r.suggestion, 0) + 1
        
        return {
            "concept_count": len(reports),
            "avg_resistance": round(avg_resistance, 4),
            "avg_entropy": round(avg_entropy, 4),
            "avg_drift_cost": round(avg_drift, 4),
            "avg_total_cost": round(avg_total, 4),
            "suggestions": suggestions
        }
