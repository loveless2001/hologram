# hologram/gravity.py
import numpy as np
import hashlib
import threading
import faiss
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import os

try:
    import psutil
except ImportError:
    psutil = None

# --- Streaming Log ---
def log_event(event_type: str, message: str, details: Dict = None):
    """Emit a streaming log event."""
    # ANSI colors
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    
    color = RESET
    if event_type == "FUSION": color = GREEN
    elif event_type == "MITOSIS": color = CYAN
    elif event_type == "GRAVITY": color = YELLOW
    elif event_type == "ERROR": color = RED
    
    print(f"{color}[{event_type}] {message}{RESET}")
    if details:
        print(f"{color}      └─ {details}{RESET}")

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

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

def pca2(X: np.ndarray):
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2)), np.eye(2), np.zeros((2,))
    
    if PCA is None:
        # Fallback to manual SVD if sklearn is missing
        mu = X.mean(axis=0, keepdims=True)
        C = X - mu
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        V2 = Vt[:2].T
        proj = C @ V2
        return proj, V2, mu.squeeze(0)
    
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)
    return proj, pca.components_.T, pca.mean_


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
# Tier constants
TIER_DOMAIN = 1  # Dynamic concepts (physics allowed)
TIER_SYSTEM = 2  # System concepts (physics forbidden)
TIER_META = 3  # Meta-operators (not stored as vectors)

@dataclass
class Concept:
    name: str
    vec: np.ndarray
    mass: float = 1.0
    count: int = 1
    negation: bool = False  # True if concept contains negation
    last_reinforced: int = 0  # Global step when last reinforced
    canonical_id: Optional[str] = None  # If set, this is an alias pointing to canonical concept
    fused_from: List[str] = field(default_factory=list)  # Track concepts that were fused into this one
    
    # NEW: 3-Tier Ontology Fields
    tier: int = TIER_DOMAIN
    project: str = "default"  # Project/domain namespace
    origin: str = "kb"  # Origin type: "kb", "runtime", "manual", "system_design"
    last_mitosis_step: int = -1000  # Last mitosis event step (for cooldown)
    last_fusion_step: int = -1000  # Last fusion event step (for cooldown)
    
    # Cost Engine: Track previous vector for drift calculation
    previous_vec: Optional[np.ndarray] = None
    
    # Evolution Pipeline Fields
    status: str = "active"
    age: int = 0
    original_vec: Optional[np.ndarray] = None
    vector_history: List[np.ndarray] = field(default_factory=list) # Store last 3 vectors

def is_protected_namespace(name: str) -> bool:
    """Check if concept name belongs to protected namespace."""
    protected_prefixes = [
        "system:",
        "meta:",
        "hologram:",
        "architecture:",
    ]
    return any(name.startswith(prefix) for prefix in protected_prefixes)

def can_interact(a: Concept, b: Concept) -> bool:
    """
    Check if two concepts can interact (fuse/drift).
    
    Rules:
    - Both must be Tier 1 (Domain)
    - Both must be from same project
    - Both must have same origin type
    """
    if a.tier != TIER_DOMAIN or b.tier != TIER_DOMAIN:
        return False
    if a.project != b.project:
        return False
    if a.origin != b.origin:
        return False
    return True


@dataclass
class ProbeStep:
    """A single step in the probe's drift history."""
    position: np.ndarray
    neighbors: List[Tuple[str, float]]  # [(concept_name, sim)]
    chosen: List[str]                   # ids actually used for force calc

@dataclass
class Probe:
    """A semantic probe that drifts through the Gravity Field."""
    name: str
    vec: np.ndarray          # current position
    mass: float = 0.1
    history: List[ProbeStep] = field(default_factory=list)
    previous_vec: Optional[np.ndarray] = None # For inertia calculation
    
    # Legacy fields support (optional, can be deprecated)
    # trajectory: List[np.ndarray] = field(default_factory=list) 
    # But new design uses history[].position

    @property
    def pos(self) -> np.ndarray:
        return self.vec
        
    @property
    def velocity(self) -> np.ndarray:
        if self.previous_vec is None:
            return np.zeros_like(self.vec)
        return self.vec - self.previous_vec

@dataclass
class RetrievalNode:
    id: str                    # concept or glyph id, or "probe"
    kind: str                  # "probe" | "concept" | "glyph"
    mass: float
    score: float               # e.g. similarity to final probe position
    children: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

@dataclass
class RetrievalTree:
    root_id: str
    nodes: Dict[str, RetrievalNode]  # id -> node


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
    
    # PCA Cache
    _pca_cache: Optional[Tuple[int, Any]] = field(default=None, repr=False)

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

    def _mutual_drift(self, new_name: str, k: int = 32):
        """
        Vectorized mutual drift:
        1. Compute cosine sim between new concept and ALL others (matrix op).
        2. Select top-k neighbors.
        3. Update relations and apply drift only to those k neighbors.
        """
        if not self.concepts:
            return

        new = self.concepts[new_name]
        names = list(self.concepts.keys())
        
        # Get matrix view of all concepts
        # Note: This creates a copy. For huge scale, maintain a persistent matrix.
        X = self.get_matrix()  # [N, D]
        v = new.vec.reshape(1, -1)  # [1, D]

        # Cosine similarity: (X . v.T) / (|X|*|v|)
        # Vectors in X and v are already normalized by add_concept/encode, 
        # but let's be safe or assume they are close enough to unit length.
        # If we trust they are normalized:
        sims = (X @ v.T).flatten()  # [N]
        
        # Identify index of self to exclude
        try:
            idx = names.index(new_name)
            sims[idx] = -2.0  # Exclude self from top-k
        except ValueError:
            pass # Should not happen if new_name is in concepts

        # Top-K selection
        # argpartition is O(N)
        k = min(k, len(names) - 1)
        if k <= 0:
            return

        top_indices = np.argpartition(-sims, k)[:k]
        
        polarity = -1.0 if new.negation else 1.0
        
        for i in top_indices:
            other_name = names[i]
            sim = float(sims[i])
            
            if sim < 0.01: continue # Ignore negligible similarity

            # 1. Update Relations
            key = (min(new_name, other_name), max(new_name, other_name))
            if key in self.relations:
                old = self.relations[key] * self.gamma_decay
                self.relations[key] = (old + polarity * sim) / 2.0
            else:
                self.relations[key] = polarity * sim
            
            # 2. Apply Drift
            other = self.concepts[other_name]
            
            # Direction: other -> new (attraction) or new -> other (repulsion?)
            # Original logic: 
            # direction = (new.vec - other.vec)
            # other.vec += polarity * step * direction
            # new.vec -= polarity * step * direction
            
            # Re-compute direction vector (can't easily vectorize this part with in-place updates without more memory)
            direction = new.vec - other.vec
            dist = np.linalg.norm(direction) + 1e-8
            direction /= dist
            
            step = self.eta * sim
            
            if abs(step) < self.quantization_level:
                continue

            # Apply forces
            other.vec += polarity * step * direction
            new.vec -= polarity * step * direction
            
            # Normalize immediately to keep stability
            other.vec /= (np.linalg.norm(other.vec) + 1e-8)
        
        # Final normalize for new concept
        new.vec /= (np.linalg.norm(new.vec) + 1e-8)

    def check_mitosis(self, name: str, threshold: float = None, 
                      mass_threshold: float = None, 
                      cooldown_steps: int = None) -> bool:
        """
        Check if a concept is under semantic tension and needs splitting.
        Returns True if mitosis occurred.
        """
        from .config import Config
        if threshold is None: threshold = Config.gravity.MITOSIS_THRESHOLD
        if mass_threshold is None: mass_threshold = Config.gravity.MITOSIS_MASS_THRESHOLD
        if cooldown_steps is None: cooldown_steps = Config.gravity.COOLDOWN_STEPS
        with self._lock:
            if name not in self.concepts:
                return False
                
            concept = self.concepts[name]
            
            # NEW: Tier validation
            if concept.tier != TIER_DOMAIN:
                return False
            
            # NEW: Mass threshold
            if concept.mass < mass_threshold:
                return False
            
            # NEW: Cooldown check
            if self.global_step - concept.last_mitosis_step < cooldown_steps:
                return False
            
            # NEW: Origin check
            if concept.origin == "system_design" or concept.origin == "code_map":
                return False
            
            # NEW: Namespace protection
            if is_protected_namespace(name):
                return False
                
            # 1. Gather neighbors
            neighbors = []
            for other_name in self.concepts:
                if other_name == name: continue
                key = (min(name, other_name), max(name, other_name))
                strength = self.relations.get(key, 0.0)
                if strength > 0.3: # Strong connection threshold
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
            log_event("MITOSIS", f"Splitting '{name}'", 
                     {"centroid_dist": f"{dist:.3f}", "mass": f"{self.concepts[name].mass:.2f}"})
            original = self.concepts[name]
            
            # Create siblings
            name1 = f"{name}_1"
            name2 = f"{name}_2"
            
            # Vectors: Position at the centroids
            vec1 = centroids[0]
            vec1 /= np.linalg.norm(vec1)
            
            vec2 = centroids[1]
            vec2 /= np.linalg.norm(vec2)
            
            # Pass tier/project/origin to siblings
            self.add_concept(name1, vec=vec1, mass=original.mass/2, negation=original.negation,
                             tier=original.tier, project=original.project, origin=original.origin)
            self.add_concept(name2, vec=vec2, mass=original.mass/2, negation=original.negation,
                             tier=original.tier, project=original.project, origin=original.origin)
            
            # Update cooldown
            if name1 in self.concepts:
                self.concepts[name1].last_mitosis_step = self.global_step
            if name2 in self.concepts:
                self.concepts[name2].last_mitosis_step = self.global_step

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

    def neighborhood_divergence(self, a_name: str, b_name: str, threshold: float = 0.4) -> float:
        """
        Check if two concepts have diverging neighborhoods.
        Returns divergence score (higher = more divergent).
        """
        a_neighbors = set()
        b_neighbors = set()
        
        for (n1, n2), strength in self.relations.items():
            if strength > 0.3:
                # Exclude the other concept from neighbors list to focus on shared 3rd parties
                if n1 == a_name and n2 != b_name:
                    a_neighbors.add(n2)
                elif n2 == a_name and n1 != b_name:
                    a_neighbors.add(n1)
                
                if n1 == b_name and n2 != a_name:
                    b_neighbors.add(n2)
                elif n2 == b_name and n1 != a_name:
                    b_neighbors.add(n1)
        
        if not a_neighbors or not b_neighbors:
            return 0.0
        
        # Jaccard distance
        intersection = len(a_neighbors & b_neighbors)
        union = len(a_neighbors | b_neighbors)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        return 1.0 - jaccard

    def check_fusion_all(self, base_threshold: float = 0.85, cooldown_steps: int = 10):
        """
        Scan for concepts that are close enough to fuse.
        Uses FAISS for efficiency.
        
        Calibration:
        - Massive concepts act like black holes (lower fusion threshold).
        - Threshold = Base - (log(Mass) * 0.02)
        """
        if len(self.concepts) < 2:
            return 0
            
        # 1. Build temporary index for current state
        # (Optimization: Maintain a persistent index in Gravity class if scale increases)
        d = self.dim
        index = faiss.IndexFlatIP(d)
        
        names = list(self.concepts.keys())
        # Filter out aliases/ghosts AND check tier constraints
        active_names = [
            n for n in names 
            if (self.concepts[n].canonical_id is None and 
                self.concepts[n].tier == TIER_DOMAIN and
                not is_protected_namespace(n) and
                self.global_step - self.concepts[n].last_fusion_step > cooldown_steps)
        ]
        
        if len(active_names) < 2:
            return 0
            
        vecs = np.stack([self.concepts[n].vec for n in active_names]).astype('float32')
        index.add(vecs)
        
        # 2. Search for neighbors
        # k=2 because nearest is always self
        D, I = index.search(vecs, 2)
        
        fused_count = 0
        processed = set()
        
        for i, (distances, indices) in enumerate(zip(D, I)):
            name_a = active_names[i]
            if name_a in processed: continue
            
            # Identify neighbor (handle case where self is not first)
            idx_b = -1
            sim = 0.0
            
            if indices[0] != i:
                idx_b = indices[0]
                sim = distances[0]
            elif indices[1] != i:
                idx_b = indices[1]
                sim = distances[1]
                
            if idx_b == -1: continue # No valid neighbor found
            
            name_b = active_names[idx_b]
            if name_b in processed: continue
            
            # NEW: Tier interaction check
            if not can_interact(self.concepts[name_a], self.concepts[name_b]):
                continue
                
            # NEW: Neighborhood divergence check
            if self.neighborhood_divergence(name_a, name_b) > 0.6:
                continue

            # NEW: Code Concept Protection
            # Code concepts rely on precise file/span definitions.
            # We prevent fusion of code concepts with anything to preserve their identity.
            if self.concepts[name_a].origin == "code_map" or self.concepts[name_b].origin == "code_map":
                continue

            sim = distances[1]
            
            # 3. Calibrate Threshold based on Mass (Black Hole Effect)
            mass_a = self.concepts[name_a].mass
            mass_b = self.concepts[name_b].mass
            max_mass = max(mass_a, mass_b)
            
            # Larger mass = stronger pull = lower threshold needed to capture
            calibrated_threshold = base_threshold - (np.log1p(max_mass) * 0.02)
            calibrated_threshold = max(0.6, calibrated_threshold) # Safety floor
            
            if sim > calibrated_threshold:
                # Fuse!
                # Determine canonical (larger mass wins)
                if mass_a >= mass_b:
                    canonical, variant = name_a, name_b
                else:
                    canonical, variant = name_b, name_a
                
                log_event("FUSION", f"'{variant}' absorbed by '{canonical}'", 
                         {"sim": f"{sim:.3f}", "thresh": f"{calibrated_threshold:.3f}", "mass_gain": f"+{self.concepts[variant].mass:.2f}"})
                
                self.fuse_concepts(variant, canonical, transfer_mass=True)
                
                # Update cooldown
                if canonical in self.concepts:
                    self.concepts[canonical].last_fusion_step = self.global_step
                
                processed.add(variant)
                fused_count += 1
                
        return fused_count

    def step_dynamics(self):
        """
        Run one step of the dynamic physics simulation.
        - Auto-Fusion (Gravity)
        - Auto-Mitosis (Cell Division)
        """
        with self._lock:
            # 1. Fusion (Pull together)
            n_fused = self.check_fusion_all()
            
            # 2. Mitosis (Split apart)
            # Check random subset or high-tension candidates to save compute
            # For now, check top 10 massive concepts (most likely to have tension)
            sorted_concepts = sorted(
                [n for n, c in self.concepts.items() if c.canonical_id is None], 
                key=lambda n: self.concepts[n].mass, 
                reverse=True
            )[:10]
            
            n_split = 0
            for name in sorted_concepts:
                if self.check_mitosis(name):
                    n_split += 1
            
            if n_fused > 0 or n_split > 0:
                log_event("GRAVITY", "System Equilibrium Adjusted", {"fused": n_fused, "split": n_split})

    def fuse_concepts(self, variant_name: str, canonical_name: str, transfer_mass: bool = True):
        """
        Fuse (merge) a variant concept into a canonical concept.
        This is the reverse of mitosis - concept fusion.
        
        Args:
            variant_name: The variant/alias concept to merge
            canonical_name: The canonical concept to merge into
            transfer_mass: If True, transfer mass from variant to canonical
        
        Returns:
            True if fusion occurred, False otherwise
        """
        with self._lock:
            if variant_name not in self.concepts:
                return False
            
            if variant_name == canonical_name:
                return False # Prevent self-fusion
            
            if canonical_name not in self.concepts:
                # If canonical doesn't exist, promote variant to canonical
                variant = self.concepts[variant_name]
                variant.name = canonical_name
                self.concepts[canonical_name] = variant
                del self.concepts[variant_name]
                return True
            
            variant = self.concepts[variant_name]
            canonical = self.concepts[canonical_name]
            
            print(f"[Fusion] Merging '{variant_name}' → '{canonical_name}' (mass: {variant.mass:.2f} + {canonical.mass:.2f})")
            
            # Transfer mass
            if transfer_mass:
                canonical.mass += variant.mass
                canonical.count += variant.count
            
            # Track fusion history
            canonical.fused_from.append(variant_name)
            
            # Convert variant to alias (pointer)
            variant.canonical_id = canonical_name
            variant.mass = 0.0  # Ghost node, no gravitational pull
            
            # Transfer relations
            for (n1, n2), strength in list(self.relations.items()):
                if variant_name in (n1, n2):
                    other_name = n2 if n1 == variant_name else n1
                    
                    if other_name == canonical_name:
                        # Self-reference, remove
                        del self.relations[(n1, n2)]
                        continue
                    
                    # Create new relation with canonical
                    new_key = (min(canonical_name, other_name), max(canonical_name, other_name))
                    
                    if new_key in self.relations:
                        # Merge strengths (weighted average)
                        self.relations[new_key] = (self.relations[new_key] + strength) / 2.0
                    else:
                        self.relations[new_key] = strength
                    
                    # Remove old relation
                    del self.relations[(n1, n2)]
            
            canonical.last_reinforced = self.global_step
            
            return True
    
    def resolve_pronoun(self, sentence: str, pronoun_span: str) -> Optional[str]:
        """
        Resolve a pronoun using vector similarity in the gravity field.
        Fallback mechanism when structural coref fails.
        """
        if not self.concepts:
            return None
            
        # 1. Encode the sentence context
        # We use the whole sentence vector to find the "topic"
        sent_vec = self.encode(sentence)
        
        # 2. Find nearest concept
        # We want the concept that is most "attracted" to this sentence context
        # This is effectively finding the topic of the sentence
        
        best_match = None
        best_sim = -1.0
        
        # Optimization: Use FAISS if available via get_matrix, but for now iterate
        # We only check Domain concepts (Tier 1)
        
        for name, concept in self.concepts.items():
            if concept.tier != TIER_DOMAIN:
                continue
            if concept.canonical_id: # Skip aliases
                continue
                
            sim = cosine(sent_vec, concept.vec)
            
            # Boost by mass? Heavier concepts are more likely topics?
            # The user suggested: "GravityField.nearest(v_sentence) -> best_match"
            # And "Gravity fallback picks the heavier attractor"
            
            # Let's use a gravity-weighted score: sim * log(mass)
            score = sim * np.log1p(concept.mass)
            
            if score > best_sim:
                best_sim = score
                best_match = name
                
        if best_match:
            log_event("COREF-FALLBACK", f"'{pronoun_span}' -> '{best_match}'", {"score": f"{best_sim:.3f}"})
            
        return best_match

    def resolve_canonical(self, name: str) -> str:
        """
        Follow the canonical_id chain to find the true canonical concept.
        
        Args:
            name: Concept name to resolve
        
        Returns:
            Canonical concept name
        """
        visited = set()
        current = name
        
        while current in self.concepts:
            if current in visited:
                # Circular reference, break
                return current
            
            visited.add(current)
            concept = self.concepts[current]
            
            if concept.canonical_id is None:
                # Found canonical
                return current
            
            # Follow pointer
            current = concept.canonical_id
        
        return name
    
    def is_alias(self, name: str) -> bool:
        """
        Check if a concept is an alias (has canonical_id set).
        
        Args:
            name: Concept name to check
        
        Returns:
            True if alias, False if canonical or not found
        """
        if name not in self.concepts:
            return False
        return self.concepts[name].canonical_id is not None

    def add_concept(
        self, 
        name: str, 
        text: Optional[str] = None, 
        vec: Optional[np.ndarray] = None, 
        mass: float = 1.0, 
        negation: Optional[bool] = None, 
        is_glyph: bool = False,
        # NEW parameters
        tier: int = TIER_DOMAIN,
        project: str = "default",
        origin: str = "kb"
    ):
        with self._lock:
            self.global_step += 1
            
            if name in self.concepts:
                # Reinforcement: restore mass and update timestamp
                c = self.concepts[name]
                c.count += 1
                
                if is_glyph:
                    # Glyphs are defined by their current centroid; overwrite
                    if vec is not None:
                        c.previous_vec = c.vec.copy()  # Track for drift cost
                        c.vec = vec / (np.linalg.norm(vec) + 1e-8)
                    c.mass = mass
                else:
                    # Regular concept: blend in new vector + mass
                    if vec is not None:
                        c.previous_vec = c.vec.copy()  # Track for drift cost
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
                last_reinforced=self.global_step,
                # NEW fields
                tier=tier,
                project=project,
                origin=origin,
                last_mitosis_step=-1000,
                last_fusion_step=-1000
            )
            
            # Only apply drift if Tier 1
            if tier == TIER_DOMAIN:
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
        
        # Check cache
        # "Only recompute when number of concepts jumps by >N (say 50)."
        should_recompute = True
        if self._pca_cache:
            last_n, last_res = self._pca_cache
            if abs(len(self.concepts) - last_n) < 50:
                # Reuse basis (components) and mean to project CURRENT X
                # This avoids re-fitting PCA (O(N*D^2)) and only does projection (O(N*D*2))
                _, comps, mu = last_res
                
                # Project: (X - mu) @ components.T
                # comps is already V2 (shape [D, 2]), so we do X @ comps
                # Wait, pca2 returns comps as pca.components_.T which is [D, 2]
                
                # Manual projection:
                C = X - mu
                proj = C @ comps
                
                names = list(self.concepts.keys())
                return proj, names
                
        # Recompute
        proj, comps, mu = pca2(X)
        self._pca_cache = (len(self.concepts), (proj, comps, mu))
        
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
                    "last_reinforced": c.last_reinforced,
                    "canonical_id": c.canonical_id,
                    "fused_from": c.fused_from,
                    # NEW fields
                    "tier": c.tier,
                    "project": c.project,
                    "origin": c.origin,
                    "last_mitosis_step": c.last_mitosis_step,
                    "last_fusion_step": c.last_fusion_step,
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
                # Migration: default tier=1 for old saves
                tier = data.get("tier", TIER_DOMAIN)
                project = data.get("project", "default")
                origin = data.get("origin", "kb")
                last_mitosis_step = data.get("last_mitosis_step", -1000)
                last_fusion_step = data.get("last_fusion_step", -1000)

                self.concepts[name] = Concept(
                    name=name,
                    vec=np.array(data["vec"], dtype=np.float32),
                    mass=data["mass"],
                    count=data["count"],
                    negation=data.get("negation", False),
                    last_reinforced=data.get("last_reinforced", 0),
                    canonical_id=data.get("canonical_id"),
                    fused_from=data.get("fused_from", []),
                    tier=tier,
                    project=project,
                    origin=origin,
                    last_mitosis_step=last_mitosis_step,
                    last_fusion_step=last_fusion_step,
                )

            # Restore relations
            self.relations = {}
            for k_str, v in state.get("relations", {}).items():
                parts = k_str.split("|")
                if len(parts) == 2:
                    self.relations[(parts[0], parts[1])] = v



    # --- Probe Physics ---
    def nearest_concepts(self, vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find nearest concepts using mass-weighted similarity.
        sim = cosine(vec, c.vec) * log(1 + c.mass)
        """
        if not self.concepts:
            return []
            
        # Filter for valid attractors (Tier 1 Domain or Glyphs)
        # Optimization: Maintain a separate list/matrix for these
        candidates = [
            n for n in self.concepts.keys() 
            if self.concepts[n].tier == TIER_DOMAIN or self.concepts[n].tier == TIER_META 
            or n.startswith("glyph:")
        ]
        
        if not candidates:
            return []

        # Brute force for physics accuracy (taking mass into account)
        # FAISS search is purely geometric (cosine), so we'd need to re-rank.
        
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        
        results = []
        for name in candidates:
            c = self.concepts[name]
            # Geometric similarity
            cos_sim = float(np.dot(vec, c.vec))
            
            # Mass-weighted similarity
            # High mass nodes pull from further away
            weighted_sim = cos_sim * np.log1p(c.mass)
            
            results.append((name, weighted_sim))
            
        # Sort by weighted score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def step_probe(
        self,
        probe: Probe,
        top_k: int = 10,
        min_sim: float = 0.2, # Minimum weighted similarity to attract
        alpha: float = 0.4,   # Drift rate (0.3 - 0.6)
        beta: float = 0.15,   # Inertia/Damping (0.1 - 0.2)
    ) -> Probe:
        """
        Execute one physics step for the probe.
        new_vec = normalize(old_vec + alpha * attraction - beta * inertia)
        """
        # 1. Find massive neighbors
        neighbors = self.nearest_concepts(probe.vec, top_k=top_k)
        
        # Filter by minimum attraction
        used = [(name, sim) for name, sim in neighbors if sim >= min_sim]
        
        if not used:
            # No attraction, just drift or stop?
            # Record step with no movement
            probe.history.append(ProbeStep(
                position=probe.vec.copy(),
                neighbors=neighbors,
                chosen=[]
            ))
            return probe

        # 2. Accumulate Attraction Force
        attraction = np.zeros_like(probe.vec, dtype=np.float32)
        total_weight = 0.0

        for name, weight in used:
            c = self.concepts[name]
            # Weight already includes mass factor from nearest_concepts
            # But standard gravity is Force ~ Mass * Direction.
            # Here 'weight' is the scalar magnitude of importance.
            
            # Direction vector
            direction = c.vec - probe.vec
            # Normalize direction? 
            # Or assume c.vec and probe.vec are unit, direction length matters?
            # Standard centroid logic: weighted sum of vectors.
            
            w = max(weight, 0.0)
            total_weight += w
            attraction += w * (c.vec - probe.vec) # Pull towards concept

        if total_weight > 0:
             attraction /= (total_weight + 1e-8)

        # 3. Calculate Inertia (Velocity)
        inertia = np.zeros_like(probe.vec)
        if probe.previous_vec is not None:
            inertia = probe.vec - probe.previous_vec

        # 4. Update Position
        # new_vec = normalize(old_vec + alpha * attraction - beta * inertia)
        # Note: Inertia usually *keeps* you moving in same direction. 
        # Damping *opposes* motion.
        # If we want DAMPING (friction), we subtract velocity: - beta * velocity.
        # If we want MOMENTUM, we add velocity: + beta * velocity.
        # User said: "beta ~ 0.1-0.2 reduces oscillation" -> implies damping.
        # Formula given: "- beta * inertia".
        
        new_pos = probe.vec + (alpha * attraction) - (beta * inertia)
        new_pos /= (np.linalg.norm(new_pos) + 1e-8)

        # Update state
        probe.previous_vec = probe.vec.copy()
        probe.vec = new_pos
        
        probe.history.append(ProbeStep(
            position=probe.vec.copy(), # Store NEW position
            neighbors=neighbors,
            chosen=[name for name, _ in used]
        ))
        
        return probe

    def run_probe(
        self,
        vec: np.ndarray,
        name: str = "probe",
        max_steps: int = 8,
        tol: float = 1e-3,
        top_k: int = 10,
        min_sim: float = 0.1, # Lower threshold suitable for weighted sim
    ) -> Probe:
        """Run probe simulation until convergence."""
        probe = Probe(
            name=name, 
            vec=vec.astype("float32"),
            previous_vec=None
        )
        # Initial state recording
        # probe.history.append(...) ? No, steps record the transitions.

        for _ in range(max_steps):
            prev_vec = probe.vec.copy()
            self.step_probe(probe, top_k=top_k, min_sim=min_sim)
            
            # Check convergence
            delta = np.linalg.norm(probe.vec - prev_vec)
            if delta < tol:
                break
                
        return probe

    def build_retrieval_tree(
        self, 
        probe: Probe, 
        radius: float = 0.65,
        max_concepts: int = 32
    ) -> RetrievalTree:
        """
        Construct a retrieval tree based on Gravitational Radius.
        Include concepts where cosine(c, probe_pos) > radius.
        """
        nodes: Dict[str, RetrievalNode] = {}
        root_id = probe.name
        
        # Root Node
        nodes[root_id] = RetrievalNode(
            id=root_id,
            kind="probe",
            mass=probe.mass,
            score=1.0,
            children=[],
            meta={"steps": len(probe.history)}
        )
        
        # 1. Collect all candidates visited or nearby final position
        # We scan all concepts (or FAISS search) to find those within radius of FINAL position
        # AND maybe those high-mass ones encountered along the path?
        # User said: "For each node visited: neighbors = { n | cosine(n, probe_vec) > radius }"
        # This implies we check radius at EACH step? Or just the final?
        # "This creates a layered tree... This is how you produce a semantic region"
        # Let's collect from ALL steps to capture the "Path".
        
        candidate_scores = {} # name -> max_similarity
        
        # Check neighborhoods of all history points
        for step in probe.history:
            # We can use the neighbors found during step (mass-weighted)
            # Or re-scan for geometric radius?
            # The user specified: "neighbors = { n | cosine(n, probe_vec) > radius }"
            # This is GEOMETRIC cosine, not mass-weighted for the *Structure*.
            
            # Optimization: Use FAISS search for radius query
            # But we are inside Gravity (no FAISS index guarantee).
            # We'll use brute force on the active set if small, or reuse step neighbors.
            
            pos = step.position
            
            for name, c in self.concepts.items():
                if name.startswith("system:"): continue # Optional
                
                sim = cosine(pos, c.vec)
                if sim > radius:
                    candidate_scores[name] = max(candidate_scores.get(name, 0.0), sim)

        # Sort by similarity
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:max_concepts]
        
        # Build Nodes
        for name, sim in top_candidates:
            c = self.concepts[name]
            kind = "glyph" if name.startswith("glyph:") else "concept"
            
            node = RetrievalNode(
                id=name,
                kind=kind,
                mass=c.mass,
                score=sim,
                children=[],
                meta={"count": c.count}
            )
            nodes[name] = node
            
            # Attach to root for now (flat tree? or hierarchical?)
            # "Query - Concept A - Concept A2..."
            # Simple version: All connect to Root.
            # Better version: Connect to nearest parent?
            # For Phase 4 MVP, flat connection to Probe Root is acceptable, 
            # effectively a "Galaxy" around the probe.
            nodes[root_id].children.append(name)
            
        return RetrievalTree(root_id=root_id, nodes=nodes)


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

    @property
    def concepts(self):
        return self.sim.concepts
        
    @property
    def relations(self):
        return self.sim.relations



    def add(
        self, 
        name: str, 
        vec: np.ndarray, 
        # NEW parameters
        tier: int = TIER_DOMAIN,
        project: str = "default",
        origin: str = "kb"
    ):
        vec = vec.astype("float32")
        vec /= (np.linalg.norm(vec) + 1e-8)
        self.sim.add_concept(
            name, 
            vec=vec,
            tier=tier,
            project=project,
            origin=origin
        )
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
        """Spawn a new probe. Wraps Gravity.run_probe logic but initializes object."""
        # This was legacy. Now we let Gravity.run_probe create it.
        # But to maintain API slightly -> return a fresh Probe object
        vec = vec.astype("float32")
        vec /= (np.linalg.norm(vec) + 1e-8)
        return Probe(name=f"probe:{abs(hash(str(vec.data)))}", vec=vec)

    def simulate_trajectory(self, probe: Probe, steps: int = 5) -> Probe:
        """
        Simulate probe drift.
        Delegates to Gravity.step_probe.
        """
        for _ in range(steps):
             self.sim.step_probe(probe)
        return probe

    def check_mitosis(self, name: str) -> bool:
        """Wrapper for Gravity.check_mitosis."""
        return self.sim.check_mitosis(name)

    def resolve_pronoun(self, sentence: str, pronoun_span: str) -> Optional[str]:
        """Wrapper for Gravity.resolve_pronoun."""
        return self.sim.resolve_pronoun(sentence, pronoun_span)

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
