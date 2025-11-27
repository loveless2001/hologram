# hologram/smi.py
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json

@dataclass
class MemoryPacket:
    """
    A structured snapshot of the holographic field for LLM consumption.
    Represents the 'Symbolic Memory Interface' (SMI).
    """
    seed: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    glyphs: List[Dict[str, Any]]
    trajectory_steps: int = 0
    field_stats: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_prompt_block(self) -> str:
        """
        Convert the packet into a dense, LLM-readable prompt block.
        """
        lines = [
            "### HOLOGRAPHIC MEMORY FIELD ###",
            f"Query Seed: '{self.seed}'",
            f"Trajectory Length: {self.trajectory_steps} steps",
            "",
            "#### NODES (Concepts)",
        ]
        
        if not self.nodes:
            lines.append("(No relevant concepts found)")
        else:
            for n in self.nodes:
                age = n.get('age', '?')
                mass = n.get('mass', 0.0)
                lines.append(f"- {n['name']} (mass={mass}, age={age})")

        lines.append("")
        lines.append("#### EDGES (Relations)")
        if not self.edges:
            lines.append("(No strong relations)")
        else:
            for e in self.edges:
                rel = e.get('relation', 0.0)
                ten = e.get('tension', 0.0)
                lines.append(f"- {e['a']} <-> {e['b']} (rel={rel}, tension={ten})")

        lines.append("")
        lines.append("#### GLYPHS (Anchors)")
        if not self.glyphs:
            lines.append("(No nearby anchors)")
        else:
            for g in self.glyphs:
                lines.append(f"- {g['id']} (mass={g['mass']}, sim={g['similarity']})")

        return "\n".join(lines)
