# hologram/gravity.py
import numpy as np
import hashlib
import threading
import faiss
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os

try:
    import psutil
except ImportError:
    psutil = None

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
STOPWORDS = set("a an the of to in for on at by and or if is are was were be been being as from with without than".split())
NEGATION_WORDS = set(["not", "no", "never", "n't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't", "doesnt", "doesn't", "didnt", "didn't", "wont", "won't", "wouldnt", "wouldn't", "cant", "can't", "couldnt", "couldn't", "shouldnt", "shouldn't"])

def detect_negation(text: str) -> bool:
    """Detect if text contains negation markers."""
    if not text:
        return False
    text_lower = text.lower()
    
    # Check for common negation patterns in the original text
    if any(word in text_lower for word in ["n't", "not ", " no ", "never "]):
        return True
    
    # Also check tokenized version
    tokens = text_lower.replace("'", "").replace("'", "").split()
    return any(tok in NEGATION_WORDS for tok in tokens)


def calibrate_quantization() -> float:
    """
    Calibrate quantization level based on hardware specs.
    Lower level = more propagation (better hardware).
    Higher level = less propagation (weaker hardware).
    """
    # Base level (conservative)
    q_level = 0.05
    
    # 1. GPU Check (FAISS)
    try:
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            q_level -= 0.02  # Significant boost for GPU
    except Exception:
        pass

    # 2. CPU Check
    try:
        cpu_count = os.cpu_count() or 1
        if cpu_count >= 8:
            q_level -= 0.01
        elif cpu_count <= 2:
            q_level += 0.02
    except Exception:
        pass

    # 3. RAM Check (if psutil available)
    if psutil:
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            if total_gb >= 16:
                q_level -= 0.01
            elif total_gb <= 4:
                q_level += 0.02
        except Exception:
            pass

    return max(0.001, min(q_level, 0.2))  # Clamp between 0.001 and 0.2


# --- Core simulation (Memory Gravity) ---
@dataclass
class Concept:
    name: str
    vec: np.ndarray
    mass: float = 1.0
    count: int = 1
    negation: bool = False  # True if concept contains negation
    last_reinforced: int = 0  # Global step when last reinforced


@dataclass
class Probe:
    """A semantic probe that drifts through the Gravity Field."""
    pos: np.ndarray
    trajectory: List[np.ndarray] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(0))  # Will be init to dim

    def __post_init__(self):
        if not self.trajectory:
            self.trajectory.append(self.pos.copy())

    def step(self, gravity: 'Gravity', step_size: float = 0.1):
        """
        Perform one step of drift based on gravitational pull of concepts.
        p_{t+1} = norm(p_t + sum(sim(p_t, c_i) * m_i * unit(c_i - p_t)))
        """
        force = np.zeros_like(self.pos)
        
        # Calculate aggregate force from all concepts
        # Optimization: In a real large-scale system, we'd use FAISS to find nearest neighbors first
        # For now, we iterate all (or use a random subset if too large)
        
        # We can use matrix operations if we have gravity.get_matrix()
        # But we need masses too.
        
        for name, concept in gravity.concepts.items():
            # Direction: concept -> probe (attraction pulls probe TO concept)
            # vector from probe to concept
            direction = concept.vec - self.pos
            dist = np.linalg.norm(direction) + 1e-8
            unit_dir = direction / dist
            
            # Similarity (Cosine) acts as a "relevance gate"
            # If they are orthogonal, pull is weak even if mass is high?
            # Gravity usually depends on distance (1/r^2). 
            # In cosine space, distance is (1-cos).
            # Let's use the formula: Force ~ Mass * Similarity * Direction
            
            sim = cosine(self.pos, concept.vec)
            
            if sim < 0.1: continue # Ignore distant concepts
            
            # Pull = Mass * Sim
            pull = concept.mass * sim
            
            force += pull * unit_dir
            
        # Update position
        self.pos += step_size * force
        self.pos /= (np.linalg.norm(self.pos) + 1e-8)
        self.trajectory.append(self.pos.copy())


@dataclass
class Gravity:
    dim: int = 256
    eta: float = 0.05
    alpha_neg: float = 0.2
    gamma_decay: float = 0.98
    seed: int = 13
    isolation_drift: float = 0.01  # Drift rate for unreinforced concepts
    mass_decay: float = 0.95  # Mass decay for unreinforced concepts
    quantization_level: Optional[float] = None  # Minimum action threshold (Planck constant)

    concepts: Dict[str, Concept] = field(default_factory=dict)
    relations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    global_step: int = 0  # Increments with each concept addition

    def __post_init__(self):
        self._lock = threading.RLock()  # Thread safety for all mutations
        if self.quantization_level is None:
            self.quantization_level = calibrate_quantization()

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
            
            # RECORD RELATION BEFORE DRIFT (preserves original semantic distance)
            # This ensures relations reflect true semantic similarity, not post-drift convergence
            polarity = -1.0 if new.negation else 1.0
            key = (min(name, new_name), max(name, new_name))
            if key in self.relations:
                old = self.relations[key] * self.gamma_decay
                self.relations[key] = (old + polarity * sim) / 2.0
            else:
                self.relations[key] = polarity * sim
            
            # THEN APPLY DRIFT (for retrieval mechanics)
            direction = (new.vec - other.vec)
            dist = np.linalg.norm(direction) + 1e-8
            direction /= dist
            step = self.eta * sim
            
            # QUANTIZATION: Check if the interaction is strong enough to propagate
            # This acts like a Planck constant for the gravity field
            if abs(step) < self.quantization_level:
                continue
            
            # NEGATION HANDLING: Reverse pull if new concept has negation
            # This makes negated concepts repel instead of attract
            other.vec += polarity * step * direction
            new.vec -= polarity * step * direction
            other.vec /= (np.linalg.norm(other.vec) + 1e-8)
            new.vec /= (np.linalg.norm(new.vec) + 1e-8)

    def check_mitosis(self, name: str, threshold: float = 0.5) -> bool:
        """
        Check if a concept is under semantic tension and needs splitting.
        Returns True if mitosis occurred.
        """
        with self._lock:
            if name not in self.concepts:
                return False
                
            # 1. Gather neighbors
            neighbors = []
            for other_name in self.concepts:
                if other_name == name: continue
                key = (min(name, other_name), max(name, other_name))
                if self.relations.get(key, 0.0) > 0.3: # Strong connection threshold
                    neighbors.append(other_name)
            
            if len(neighbors) < 3:
                return False
                
            # 2. Cluster neighbors using FAISS K-means (Geometry-Based)
            vecs = np.stack([self.concepts[n].vec for n in neighbors]).astype('float32')
            d = vecs.shape[1]
            
            # Use FAISS K-means for robust geometric clustering
            # niter=20, nredo=1 for speed/quality balance
            # min_points_per_centroid=1 to allow splitting small clusters
            kmeans = faiss.Kmeans(d, k=2, niter=20, nredo=1, seed=self.seed, min_points_per_centroid=1)
            kmeans.train(vecs)
            
            # Check distance between centroids
            centroids = kmeans.centroids
            dist = 1.0 - cosine(centroids[0], centroids[1])
            
            if dist < threshold: # Centroids are too close, no split needed
                return False
                
            # Assign neighbors to clusters
            # index.search returns (distances, indices)
            D, I = kmeans.index.search(vecs, 1)
            
            cluster1 = []
            cluster2 = []
            
            for i, cluster_idx in enumerate(I.flatten()):
                if cluster_idx == 0:
                    cluster1.append(neighbors[i])
                else:
                    cluster2.append(neighbors[i])
            
            if not cluster1 or not cluster2:
                return False
                
            # 3. Perform Mitosis
            print(f"[Mitosis] Splitting '{name}' into '{name}_1' and '{name}_2' (Centroid Dist: {dist:.3f})")
            original = self.concepts[name]
            
            # Create siblings
            name1 = f"{name}_1"
            name2 = f"{name}_2"
            
            # Vectors: Position at the centroids
            vec1 = centroids[0]
            vec1 /= np.linalg.norm(vec1)
            
            vec2 = centroids[1]
            vec2 /= np.linalg.norm(vec2)
            
            self.add_concept(name1, vec=vec1, mass=original.mass/2, negation=original.negation)
            self.add_concept(name2, vec=vec2, mass=original.mass/2, negation=original.negation)
            
            # Re-link neighbors
            for n in cluster1:
                old_key = (min(name, n), max(name, n))
                strength = self.relations.get(old_key, 0.0)
                new_key = (min(name1, n), max(name1, n))
                self.relations[new_key] = strength
                
            for n in cluster2:
                old_key = (min(name, n), max(name, n))
                strength = self.relations.get(old_key, 0.0)
                new_key = (min(name2, n), max(name2, n))
                self.relations[new_key] = strength
                
            # Bridge Link (Soft Split)
            bridge_key = (min(name1, name2), max(name1, name2))
            self.relations[bridge_key] = 0.15 # Weak bridge
            
            # Cleanup original
            del self.concepts[name]
            # Clean relations involving original
            keys_to_del = [k for k in self.relations if name in k]
            for k in keys_to_del:
                del self.relations[k]
                
            return True

    def add_concept(self, name: str, text: Optional[str] = None, vec: Optional[np.ndarray] = None, mass: float = 1.0, negation: Optional[bool] = None, is_glyph: bool = False):
        with self._lock:
            self.global_step += 1
            
            if name in self.concepts:
                # Reinforcement: restore mass and update timestamp
                c = self.concepts[name]
                c.count += 1
                
                if is_glyph:
                    # Glyphs are defined by their current centroid; overwrite
                    if vec is not None:
                        c.vec = vec / (np.linalg.norm(vec) + 1e-8)
                    c.mass = mass
                else:
                    # Regular concept: blend in new vector + mass
                    if vec is not None:
                        v = vec / (np.linalg.norm(vec) + 1e-8)
                        total_mass = c.mass + mass
                        c.vec = (c.vec * c.mass + v * mass) / total_mass
                        c.vec /= (np.linalg.norm(c.vec) + 1e-8)
                        c.mass = total_mass
                    else:
                        c.mass = min(c.mass + mass, mass * 2.0)  # Cap at 2x base mass
                
                c.last_reinforced = self.global_step
                return
            
            vec = vec if vec is not None else self.encode(text or name)
            
            # Auto-detect negation if not explicitly provided
            if negation is None and text:
                negation = detect_negation(text)
            elif negation is None:
                negation = False
                
            self.concepts[name] = Concept(
                name=name, 
                vec=vec, 
                mass=mass, 
                count=1, 
                negation=negation,
                last_reinforced=self.global_step
            )
            self._mutual_drift(name)

    def apply_isolation_decay(self, staleness_threshold: int = 5):
        """Apply decay to unreinforced concepts.
        
        Concepts that haven't been reinforced recently:
        - Drift away from the centroid (isolation)
        - Lose mass (influence)
        """
        if len(self.concepts) < 2:
            return
        
        # Calculate centroid of all concepts
        centroid = np.mean([c.vec for c in self.concepts.values()], axis=0)
        
        for name, concept in self.concepts.items():
            staleness = self.global_step - concept.last_reinforced
            
            if staleness > staleness_threshold:
                # Push away from centroid (isolation drift)
                to_centroid = centroid - concept.vec
                drift_strength = self.isolation_drift * min(staleness / 10.0, 1.0)
                
                # Move AWAY from centroid
                concept.vec -= drift_strength * to_centroid
                concept.vec /= (np.linalg.norm(concept.vec) + 1e-8)
                
                # Decay mass
                concept.mass *= self.mass_decay
    
    def step_decay(self, steps: int = 1):
        """Decay relations and apply isolation drift."""
        with self._lock:
            for _ in range(steps):
                # Decay relation strengths
                for k in list(self.relations.keys()):
                    self.relations[k] *= self.gamma_decay
                
                # Apply isolation-based drift
                self.apply_isolation_decay()

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
                    "count": c.count,
                    "negation": c.negation,
                    "last_reinforced": c.last_reinforced
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
                "seed": self.seed,
                "isolation_drift": self.isolation_drift,
                "mass_decay": self.mass_decay,
                "quantization_level": self.quantization_level,
                "global_step": self.global_step
            }
        }

    def set_state(self, state: dict):
        """Restore simulation state from dictionary."""
        with self._lock:
            # Restore params
            params = state.get("params", {})
            self.dim = params.get("dim", self.dim)
            self.eta = params.get("eta", self.eta)
            self.alpha_neg = params.get("alpha_neg", self.alpha_neg)
            self.gamma_decay = params.get("gamma_decay", self.gamma_decay)
            self.seed = params.get("seed", self.seed)
            self.isolation_drift = params.get("isolation_drift", self.isolation_drift)
            self.mass_decay = params.get("mass_decay", self.mass_decay)
            self.quantization_level = params.get("quantization_level", self.quantization_level)
            self.global_step = params.get("global_step", 0)

            # Restore concepts
            self.concepts = {}
            for name, data in state.get("concepts", {}).items():
                self.concepts[name] = Concept(
                    name=name,
                    vec=np.array(data["vec"], dtype=np.float32),
                    mass=data["mass"],
                    count=data["count"],
                    negation=data.get("negation", False),
                    last_reinforced=data.get("last_reinforced", 0)
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

    def spawn_probe(self, vec: np.ndarray) -> Probe:
        """Spawn a new probe at the given vector."""
        vec = vec.astype("float32")
        vec /= (np.linalg.norm(vec) + 1e-8)
        return Probe(pos=vec, velocity=np.zeros_like(vec))

    def simulate_trajectory(self, probe: Probe, steps: int = 5) -> Probe:
        """Simulate probe drift for N steps."""
        for _ in range(steps):
            probe.step(self.sim)
        return probe

    def check_mitosis(self, name: str) -> bool:
        """Wrapper for Gravity.check_mitosis."""
        return self.sim.check_mitosis(name)

    def get_subgraph(self, concepts: List[str]) -> List[Dict]:
        """
        Retrieve a subgraph of concepts with their relations and mass.
        Returns a list of concept objects with 'related_to' fields.
        """
        graph = []
        for name in concepts:
            if name not in self.sim.concepts:
                continue
                
            c = self.sim.concepts[name]
            node = {
                "name": name,
                "mass": round(c.mass, 3),
                "related_to": []
            }
            
            # Find relations where this concept is involved
            for other_name in concepts:
                if name == other_name:
                    continue
                    
                key = (min(name, other_name), max(name, other_name))
                if key in self.sim.relations:
                    strength = self.sim.relations[key]
                    if strength > 0.1:  # Filter weak relations
                        node["related_to"].append({
                            "name": other_name,
                            "strength": round(strength, 3)
                        })
            
            node["related_to"].sort(key=lambda x: x["strength"], reverse=True)
            graph.append(node)
            
        return graph
