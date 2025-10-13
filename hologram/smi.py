# hologram/smi.py
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .glyphs import GlyphRegistry
from .store import MemoryStore


@dataclass
class ResonanceEvent:
    glyph_id: str
    score: float
    timestamp: float
    meta: dict = field(default_factory=dict)


@dataclass
class SMIState:
    """Persistent symbolic state of the memory interface."""
    events: List[ResonanceEvent] = field(default_factory=list)
    drift_index: Dict[str, float] = field(default_factory=dict)
    last_decay_step: int = 0


class SymbolicMemoryInterface:
    """
    Symbolic Memory Interface (SMI)
    --------------------------------
    Persistent layer for symbolic drift and resonance tracking.
    """

    def __init__(
        self,
        store: MemoryStore,
        registry: GlyphRegistry,
        save_path: str = "data/smi_state.json",
        autosave: bool = True,
    ):
        self.store = store
        self.registry = registry
        self.save_path = Path(save_path)
        self.autosave = autosave
        self.gamma_decay = getattr(store.sim, "gamma_decay", 0.98)
        self.state = SMIState()

        # Attempt to restore memory from previous session
        if self.save_path.exists():
            self._load_state()
            print(f"[SMI] Loaded {len(self.state.events)} events from {self.save_path}")

    # --- Core Operations ---
    def record_resonance(self, glyph_id: str, score: float, **meta):
        now = datetime.utcnow().timestamp()
        evt = ResonanceEvent(glyph_id=glyph_id, score=score, timestamp=now, meta=meta)
        self.state.events.append(evt)

        # Update drift index (weighted by recency)
        prev = self.state.drift_index.get(glyph_id, 0.0)
        self.state.drift_index[glyph_id] = (
            prev * self.gamma_decay + score * (1 - self.gamma_decay)
        )

        if self.autosave:
            self._save_state()

    def decay(self, steps: int = 1):
        """Apply temporal decay to drift index and underlying gravity."""
        for _ in range(steps):
            for gid, val in list(self.state.drift_index.items()):
                self.state.drift_index[gid] = val * self.gamma_decay
            self.store.step_decay()
        self.state.last_decay_step += steps

        if self.autosave:
            self._save_state()

    def pulse(self, query_vec: np.ndarray):
        """
        Perform one symbolic memory pulse:
        - computes resonance
        - records drift
        - auto-saves if enabled
        """
        scores = self.registry.resonance_score(query_vec)
        for gid, s in scores.items():
            self.record_resonance(gid, s)
        if self.autosave:
            self._save_state()
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def recall_field(self, top_k: int = 5) -> List[str]:
        if not self.state.drift_index:
            return []
        sorted_items = sorted(
            self.state.drift_index.items(), key=lambda x: x[1], reverse=True
        )
        return [gid for gid, _ in sorted_items[:top_k]]

    def analyze_drift(self, threshold: float = 0.05) -> Dict[str, float]:
        """Return glyphs whose drift exceeds threshold."""
        return {
            gid: val
            for gid, val in self.state.drift_index.items()
            if abs(val) > threshold
        }

    # --- Persistence ---
    def _save_state(self):
        """Internal safe save."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "events": [asdict(e) for e in self.state.events],
            "drift_index": self.state.drift_index,
            "last_decay_step": self.state.last_decay_step,
        }
        self.save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_state(self):
        """Internal safe load."""
        data = json.loads(self.save_path.read_text(encoding="utf-8"))
        self.state.events = [ResonanceEvent(**e) for e in data.get("events", [])]
        self.state.drift_index = data.get("drift_index", {})
        self.state.last_decay_step = data.get("last_decay_step", 0)

    def save(self, path: Optional[str] = None):
        """Manual save (optional path override)."""
        if path:
            self.save_path = Path(path)
        self._save_state()

    def summary(self) -> Dict[str, int]:
        return {
            "events_logged": len(self.state.events),
            "active_glyphs": len(self.state.drift_index),
            "last_decay_step": self.state.last_decay_step,
            "autosave": self.autosave,
            "path": str(self.save_path),
        }
