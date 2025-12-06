from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple, Any
import json
from pathlib import Path

@dataclass
class SymbolMetadata:
    symbol_id: str
    qualified_name: str
    signature: str
    file_path: str
    language: str
    first_seen: str   # timestamp or revision
    last_seen: str    # timestamp or revision
    status: str       # "active", "deprecated", "revived"
    vector_hash: str  # Hash of the vector for quick change detection (optional)

class SymbolRegistry:
    """
    Persistent registry to track symbol identities across code versions.
    Maps deterministically generated symbol_id -> metadata.
    """
    def __init__(self, persistence_path: Optional[str] = None):
        self.registry: Dict[str, SymbolMetadata] = {}
        self.persistence_path = persistence_path
        if self.persistence_path:
            self.load()

    def register(self, metadata: SymbolMetadata):
        self.registry[metadata.symbol_id] = metadata

    def get(self, symbol_id: str) -> Optional[SymbolMetadata]:
        return self.registry.get(symbol_id)
        
    def find_by_name(self, qualified_name: str) -> Optional[SymbolMetadata]:
        # Linear search - not efficient for huge repos but OK for now
        # Could index by name if needed
        for sym in self.registry.values():
            if sym.qualified_name == qualified_name and sym.status == "active":
                return sym
        return None

    def save(self):
        if not self.persistence_path:
            return
            
        path = Path(self.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            k: asdict(v) for k, v in self.registry.items()
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self):
        if not self.persistence_path:
            return
        
        path = Path(self.persistence_path)
        if not path.exists():
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                self.registry[k] = SymbolMetadata(**v)
        except Exception as e:
            print(f"[SymbolRegistry] Failed to load registry: {e}")
