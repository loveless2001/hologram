# hologram/cost_engine.py
"""
Cost Engine: A diagnostic meta-layer that measures cognitive effort metrics.

The Cost Engine observes the Gravity Field and Concept Graph and computes:
- Entropy: Semantic fragmentation (dispersion of neighbor vectors)
- Instability: Gradient conflict (neighbors pulling in opposing directions)
- Resistance: Inertia based on mass and connectivity
- Drift Cost: Cost to move away from current basin (or history)
- Maintenance Cost: Ongoing cost of keeping structure alive

It does NOT mutate vectors. All suggestions are advisory signals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .gravity import Gravity, cosine


@dataclass
class CostSignal:
    entropy: float
    instability: float
    resistance: float
    drift_cost: float
    maintenance_cost: float
    
    # Optional metadata for reporting
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.entropy * 0.3 +
            self.instability * 0.25 +
            self.resistance * 0.2 +
            self.drift_cost * 0.15 +
            self.maintenance_cost * 0.1
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entropy": round(self.entropy, 4),
            "instability": round(self.instability, 4),
            "resistance": round(self.resistance, 4),
            "drift_cost": round(self.drift_cost, 4),
            "maintenance_cost": round(self.maintenance_cost, 4),
            "total_cost": round(self.total, 4),
            "details": self.details
        }


class CostEngine:
    """
    Observes the GravityField and Concept Graph.
    Computes cost metrics but does not mutate state directly.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.6,
        instability_threshold: float = 0.5,
        resistance_threshold: float = 0.7,
    ):
        self.entropy_threshold = entropy_threshold
        self.instability_threshold = instability_threshold
        self.resistance_threshold = resistance_threshold

    # ---------- HELPER / ADAPTER METHODS ----------
    
    def _get_neighbors(self, field: Gravity, node_id: str) -> List[str]:
        """
        Get neighbors of a node. 
        Prioritizes Graph Relations, falls back to geometric proximity if needed.
        """
        neighbors = set()
        
        # 1. Check Relations (Graph Structure)
        for (n1, n2), strength in field.relations.items():
            if strength > 0.1:  # Threshold for valid connection
                if n1 == node_id:
                    neighbors.add(n2)
                elif n2 == node_id:
                    neighbors.add(n1)
        
        # 2. If no graph neighbors, fallback to geometric neighbors (top K)
        # This occurs if the concept is new or isolated in the graph but close in space
        if not neighbors and field.concepts:
            # Simple linear search (O(N)) - okay for diagnostics
            # For optimization, we could use a FAISS index if exposed
            concept_vec = field.concepts[node_id].vec
            candidates = []
            for name, c in field.concepts.items():
                if name == node_id: continue
                if c.canonical_id: continue # Skip aliases
                sim = cosine(concept_vec, c.vec)
                if sim > 0.3: # Minimum similarity
                    candidates.append((name, sim))
            
            # Sort and take top 5
            candidates.sort(key=lambda x: x[1], reverse=True)
            neighbors = set(n for n, s in candidates[:5])
            
        return list(neighbors)

    def _get_vector(self, field: Gravity, node_id: str) -> np.ndarray:
        return field.concepts[node_id].vec

    def _get_mass(self, field: Gravity, node_id: str) -> float:
        return field.concepts[node_id].mass

    # ---------- PUBLIC API ----------

    def evaluate_node(self, node_id: str, field: Gravity) -> CostSignal:
        """
        Evaluate cost signals for a single node/concept.
        """
        if node_id not in field.concepts:
            # Return empty/max cost signal for missing concepts
            return CostSignal(0.0, 0.0, 0.0, 0.0, 0.0, details={"error": "not_found"})

        neighbors = self._get_neighbors(field, node_id)
        vec = self._get_vector(field, node_id)
        mass = self._get_mass(field, node_id)

        entropy = self._entropy(neighbors, field)
        instability = self._instability(vec, neighbors, field)
        resistance = self._resistance(mass, neighbors)
        drift_cost = self._drift_cost(vec, neighbors, field, node_id)
        maintenance_cost = self._maintenance_cost(mass, neighbors)
        
        # Add suggestion to details
        signal = CostSignal(
            entropy=entropy,
            instability=instability,
            resistance=resistance,
            drift_cost=drift_cost,
            maintenance_cost=maintenance_cost,
        )
        
        intervention = self.should_intervene(signal)
        signal.details = {
            "neighbor_count": len(neighbors),
            "should_intervene": intervention,
            "suggestion": "intervene" if intervention else "stable"
        }
        
        return signal

    def should_intervene(self, cost: CostSignal) -> bool:
        """
        High-level decision gate.
        """
        return (
            cost.entropy > self.entropy_threshold or
            cost.instability > self.instability_threshold or
            cost.resistance > self.resistance_threshold
        )
        
    def evaluate_field(self, field: Gravity) -> Dict[str, CostSignal]:
        """Evaluate all concepts in the field."""
        reports = {}
        for name, concept in field.concepts.items():
            if concept.canonical_id is None:  # Skip aliases
                reports[name] = self.evaluate_node(name, field)
        return reports

    def field_summary(self, field: Gravity) -> Dict[str, Any]:
        """Aggregate statistics for the entire field."""
        reports = self.evaluate_field(field)
        if not reports:
            return {"count": 0}
            
        avg_total = sum(r.total for r in reports.values()) / len(reports)
        avg_entropy = sum(r.entropy for r in reports.values()) / len(reports)
        interventions = sum(1 for r in reports.values() if r.details.get("should_intervene"))
        
        return {
            "count": len(reports),
            "avg_total_cost": round(avg_total, 4),
            "avg_entropy": round(avg_entropy, 4),
            "intervention_rate": f"{interventions}/{len(reports)}",
            "percent_unstable": round(interventions / len(reports) * 100, 1)
        }

    # ---------- CORE METRICS ----------

    def _entropy(self, neighbors: List[str], field: Gravity) -> float:
        """
        Measures dispersion of neighbor vectors.
        High entropy = semantic fragmentation.
        """
        if len(neighbors) < 2:
            return 0.0

        # Gather vectors using field.concepts
        vectors = []
        for n in neighbors:
            if n in field.concepts:
                vectors.append(field.concepts[n].vec)
        
        if not vectors:
            return 0.0
            
        vectors = np.array(vectors)
        centroid = vectors.mean(axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)

        # normalized entropy proxy
        # std_dev / (mean_dist + eps)
        # If mean distance is 0 (all points identical), entropy is 0.
        mean_dist = np.mean(distances)
        if mean_dist < 1e-6:
            return 0.0
            
        return float(np.std(distances) / (mean_dist + 1e-6))

    def _instability(self, vec: np.ndarray, neighbors: List[str], field: Gravity) -> float:
        """
        Measures gradient conflict.
        If neighbors pull in opposing directions, instability is high.
        """
        if len(neighbors) < 2:
            return 0.0

        directions = []
        for n in neighbors:
            if n not in field.concepts: continue
            
            nvec = field.concepts[n].vec
            # Direction vector
            diff = nvec - vec
            norm_diff = np.linalg.norm(diff) + 1e-6
            directions.append(diff / norm_diff)

        if not directions:
            return 0.0

        directions = np.array(directions)
        mean_dir = directions.mean(axis=0)

        # variance from consensus direction
        # High variance = conflicting pulls
        angular_deviation = np.linalg.norm(directions - mean_dir, axis=1).mean()
        return float(angular_deviation)

    def _resistance(self, mass: float, neighbors: List[str]) -> float:
        """
        Resistance grows with mass and dense local connectivity.
        Represents inertia or difficulty to change.
        """
        degree = len(neighbors)
        # Logarithmic scaling to prevent runaway values
        return float(np.log1p(mass) * np.log1p(degree))

    def _drift_cost(self, vec: np.ndarray, neighbors: List[str], field: Gravity, node_id: str = None) -> float:
        """
        Cost to move away from current basin.
        User suggestion implies distance to neighbors mean? 
        Or history based drift? 
        
        User code: 
             distances = [norm(vec - nvec) for n in neighbors]
             return mean(distances)
             
        This interprets drift cost as "tension" with current neighbors.
        """
        if not neighbors:
            return 0.0

        distances = []
        for n in neighbors:
            if n in field.concepts:
                nvec = field.concepts[n].vec
                distances.append(np.linalg.norm(vec - nvec))
        
        if not distances:
            return 0.0
            
        return float(np.mean(distances))

    def _maintenance_cost(self, mass: float, neighbors: List[str]) -> float:
        """
        Ongoing cost of keeping this structure alive.
        """
        return float(mass * (1 + len(neighbors) * 0.1))
