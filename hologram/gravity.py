# hologram/gravity.py
import numpy as np
import hashlib
import faiss
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# --- Utility functions ---
def hash_embed(text: str, dim: int = 256, seed: int = 13) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    for tok in tokens if tokens else [text]:
        h = hashlib.blake2b((str(seed)+tok).encode('utf-8'), digest_size=8).digest()
        idx = int.from_bytes(h, 'little') % dim
        sign = 1 if (int.from_bytes(hashlib.blake2b((tok+'sign').encode('utf-8'), digest_size=1).digest(),'little') % 2 == 0) else -1
        vec[idx] += sign
    n = np.linalg.norm(vec) + 1e-8
    return vec / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (an * bn))

def pca2(X: np.ndarray):
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2)), np.eye(2), np.zeros((2,))
    mu = X.mean(axis=0, keepdims=True)
    C = X - mu
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    V2 = Vt[:2].T
    proj = C @ V2
    return proj, V2, mu.squeeze(0)


# --- Constants ---
STOPWORDS = set("a an the of to in for on at by and or if is are was were be been being as from with without than not no".split())


# --- Core simulation (Memory Gravity) ---
@dataclass
class Concept:
    name: str
    vec: np.ndarray
    mass: float = 1.0
    count: int = 1


@dataclass
class Gravity:
    dim: int = 256
    eta: float = 0.05
    alpha_neg: float = 0.2
    gamma_decay: float = 0.98
    seed: int = 13

    concepts: Dict[str, Concept] = field(default_factory=dict)
    relations: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def encode(self, text: str) -> np.ndarray:
        toks = text.lower().split()
        if all(t in STOPWORDS for t in toks) and len(toks) > 0:
            return np.zeros(self.dim, dtype=np.float32)
        v = hash_embed(text, dim=self.dim, seed=self.seed)
        field_mean = (
            np.mean([c.vec for c in self.concepts.values()], axis=0)
            if self.concepts
            else np.zeros(self.dim, dtype=np.float32)
        )
        v = v - self.alpha_neg * field_mean - self.alpha_neg * np.abs(v.mean())
        v /= (np.linalg.norm(v) + 1e-8)
        return v

    def _mutual_drift(self, new_name: str):
        new = self.concepts[new_name]
        for name, other in self.concepts.items():
            if name == new_name:
                continue
            sim = cosine(new.vec, other.vec)
            direction = (new.vec - other.vec)
            dist = np.linalg.norm(direction) + 1e-8
            direction /= dist
            step = self.eta * sim
            other.vec += step * direction
            new.vec -= step * direction
            other.vec /= (np.linalg.norm(other.vec) + 1e-8)
            new.vec /= (np.linalg.norm(new.vec) + 1e-8)
            key = (min(name, new_name), max(name, new_name))
            old = self.relations.get(key, 0.0) * self.gamma_decay
            self.relations[key] = (old + sim) / 2.0

    def add_concept(self, name: str, text: Optional[str] = None, vec: Optional[np.ndarray] = None, mass: float = 1.0):
        if name in self.concepts:
            c = self.concepts[name]
            c.count += 1
            c.mass += mass
            return
        vec = vec if vec is not None else self.encode(text or name)
        self.concepts[name] = Concept(name=name, vec=vec, mass=mass, count=1)
        self._mutual_drift(name)

    def step_decay(self, steps: int = 1):
        for _ in range(steps):
            for k in list(self.relations.keys()):
                self.relations[k] *= self.gamma_decay

    def get_matrix(self) -> np.ndarray:
        if not self.concepts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([c.vec for c in self.concepts.values()], axis=0)

    def project2d(self):
        X = self.get_matrix()
        proj, comps, mu = pca2(X)
        names = list(self.concepts.keys())
        return proj, names

    def get_state(self) -> dict:
        """Export current simulation state (concepts and relations)."""
        return {
            "concepts": {
                name: {
                    "vec": c.vec.tolist(),
                    "mass": c.mass,
                    "count": c.count
                }
                for name, c in self.concepts.items()
            },
            "relations": {
                f"{k[0]}|{k[1]}": v
                for k, v in self.relations.items()
            },
            "params": {
                "dim": self.dim,
                "eta": self.eta,
                "alpha_neg": self.alpha_neg,
                "gamma_decay": self.gamma_decay,
                "seed": self.seed
            }
        }

    def set_state(self, state: dict):
        """Restore simulation state from dictionary."""
        # Restore params
        params = state.get("params", {})
        self.dim = params.get("dim", self.dim)
        self.eta = params.get("eta", self.eta)
        self.alpha_neg = params.get("alpha_neg", self.alpha_neg)
        self.gamma_decay = params.get("gamma_decay", self.gamma_decay)
        self.seed = params.get("seed", self.seed)

        # Restore concepts
        self.concepts = {}
        for name, data in state.get("concepts", {}).items():
            self.concepts[name] = Concept(
                name=name,
                vec=np.array(data["vec"], dtype=np.float32),
                mass=data["mass"],
                count=data["count"]
            )

        # Restore relations
        self.relations = {}
        for k_str, v in state.get("relations", {}).items():
            parts = k_str.split("|")
            if len(parts) == 2:
                self.relations[(parts[0], parts[1])] = v


# --- Field wrapper (FAISS + Gravity) ---
class GravityField:
    def __init__(self, dim=512, use_faiss=True):
        self.dim = dim
        self.sim = Gravity(dim=dim)
        self.use_faiss = use_faiss
        self.vectors: List[np.ndarray] = []
        self.names: List[str] = []
        if use_faiss:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    def add(self, name: str, vec: np.ndarray):
        vec = vec.astype("float32")
        vec /= (np.linalg.norm(vec) + 1e-8)
        self.sim.add_concept(name, vec=vec)
        self.vectors.append(vec)
        self.names.append(name)
        if self.index is not None:
            self.index.add(np.array([vec], dtype="float32"))

    def search(self, q: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None or len(self.names) == 0:
            return []
        q = q.astype("float32")
        q /= (np.linalg.norm(q) + 1e-8)
        D, I = self.index.search(np.array([q], dtype="float32"), k)
        return [(self.names[i], float(D[0][j])) for j, i in enumerate(I[0])]

    def project2d(self):
        return self.sim.project2d()

    def step_decay(self, steps: int = 1):
        self.sim.step_decay(steps)
