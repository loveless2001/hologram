"""
hologram.probe
==============

Implements cost-aware probe traversal for knowledge retrieval.
Uses an energy-based metric to find the "lowest cognitive-effort" path
through the concept graph.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
import heapq
import numpy as np

from .gravity import Gravity, cosine
from .cost_engine import CostEngine, CostSignal

@dataclass
class RetrievalNode:
    id: str
    mass: float
    sim_to_query: float
    cost_total: float
    entropy: float
    instability: float
    resistance: float
    
    # Traversal metadata
    energy_from_query: float = 0.0  # E(v|q)
    path_energy: float = 0.0        # Cumulative energy from root
    parent_id: Optional[str] = None
    depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mass": round(self.mass, 3),
            "sim_to_query": round(self.sim_to_query, 3),
            "cost_total": round(self.cost_total, 3),
            "entropy": round(self.entropy, 3),
            "instability": round(self.instability, 3),
            "resistance": round(self.resistance, 3),
            "energy_from_query": round(self.energy_from_query, 3),
            "path_energy": round(self.path_energy, 3),
            "parent_id": self.parent_id,
            "depth": self.depth
        }

@dataclass
class RetrievalEdge:
    source: str
    target: str
    similarity: float
    relation_strength: float
    edge_energy: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "similarity": round(self.similarity, 3),
            "relation_strength": round(self.relation_strength, 3),
            "edge_energy": round(self.edge_energy, 3)
        }

@dataclass
class RetrievalTree:
    root_id: str
    nodes: Dict[str, RetrievalNode] = field(default_factory=dict)
    edges: List[RetrievalEdge] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges]
        }

class ProbeRetriever:
    """
    Simulates a probe attracted by mass/similarity but repelled by cost.
    """
    
    def __init__(self, gravity: Gravity, cost_engine: CostEngine):
        self.gravity = gravity
        self.cost_engine = cost_engine
        
        # Energy weights (node)
        self.w_s = 0.55  # Similarity weight
        self.w_m = 0.15  # Mass weight
        self.w_c = 0.30  # Cost weight
        
        # Path weights (edge)
        self.lambda_edge = 0.7  # Similarity penalty for transitions
        self.mu_edge = 0.3      # Destination cost penalty
        
        # Cache for cost signals to avoid re-computation during one retrieval
        self._cost_cache: Dict[str, CostSignal] = {}

    def _get_cost(self, node_id: str) -> CostSignal:
        if node_id not in self._cost_cache:
            self._cost_cache[node_id] = self.cost_engine.evaluate_node(node_id, self.gravity)
        return self._cost_cache[node_id]

    def _node_energy(self, node_id: str, query_vec: np.ndarray, sim_q: float) -> float:
        """
        Calculate node energy E(v|q). Lower is better.
        """
        c = self.gravity.concepts[node_id]
        cost = self._get_cost(node_id)
        
        # 1. Similarity Term (High sim -> Low energy)
        # sim is [-1, 1], we map to [0, 2] roughly? 
        # Actually standard usage: (1 - sim). 
        # If sim=1, term=0. If sim=0, term=1.
        e_sim = 1.0 - sim_q
        
        # 2. Mass Term (High mass -> Low energy)
        # 1 / log(1 + mass)
        e_mass = 1.0 / (np.log1p(c.mass) + 1e-6)
        
        # 3. Cost Term (High cost -> High energy)
        e_cost = cost.total
        
        return (
            self.w_s * e_sim +
            self.w_m * e_mass +
            self.w_c * e_cost
        )

    def _edge_energy(self, u_id: str, v_id: str, cost_v: CostSignal) -> Tuple[float, float, float]:
        """
        Calculate transition energy E_edge(u->v).
        Returns (energy, similarity, relation_strength)
        """
        vec_u = self.gravity.concepts[u_id].vec
        vec_v = self.gravity.concepts[v_id].vec
        
        sim = cosine(vec_u, vec_v)
        
        # Check explicit relation
        key = (min(u_id, v_id), max(u_id, v_id))
        relation = self.gravity.relations.get(key, 0.0)
        
        # Formula: lambda * (1-sim) + mu * cost(v).total
        e_sim = 1.0 - sim
        e_cost = cost_v.total
        
        energy = self.lambda_edge * e_sim + self.mu_edge * e_cost
        
        # Bonus for explicit relations? 
        # Maybe reduce energy if relation exists
        if relation > 0.1:
            energy *= 0.8  # 20% discount for existing strong path
            
        return energy, sim, relation

    def retrieve_tree(
        self, 
        query_vec: np.ndarray, 
        top_k_seeds: int = 100,
        max_depth: int = 2,
        final_k: int = 40
    ) -> RetrievalTree:
        """
        Execute best-first search to build a retrieval tree.
        """
        self._cost_cache.clear()
        
        # 1. Seed Candidates (Geometric Search)
        # We need a list of (node_id, sim)
        # Using simple linear scan if API not available, or gravity's generic search helper
        # Assuming gravity has direct access to concepts dict for text scan or use FAISS if needed.
        # Here we do a linear scan for simplicity or reuse internal method? 
        # Let's do linear scan on concepts as in cost_engine adapter logic, it's safer.
        
        candidates = []
        for name, c in self.gravity.concepts.items():
            if c.canonical_id: continue # Skip aliases
            if name.startswith("system:") and c.tier == 2: pass # Include system? Yes.
            
            sim = cosine(query_vec, c.vec)
            if sim > 0.1: # Minimum relevance
                candidates.append((name, sim))
                
        # Top K seeds
        candidates.sort(key=lambda x: x[1], reverse=True)
        seeds = candidates[:top_k_seeds]
        
        if not seeds:
            return RetrievalTree(root_id="QUERY")

        # 2. Priority Queue for Best-First Search
        # Item: (path_energy, node_id, parent_id, depth)
        pq = []
        
        # Track visited to avoid cycles and redundant work
        # value: best_path_energy found so far
        visited: Dict[str, float] = {}
        
        # Result containers
        # We store full Node objects as we find them
        found_nodes: Dict[str, RetrievalNode] = {}
        found_edges: List[RetrievalEdge] = []
        
        # Initialize PQ with seeds
        # Seeds are "step 1" from query? Or "step 0"? 
        # Let's treat QUERY as a virtual root. 
        # Energy(q->v) = NodeEnergy(v|q). 
        # (Since edge from q->v is implicit and pure similarity)
        
        for nid, sim in seeds:
            e_node = self._node_energy(nid, query_vec, sim)
            # Path energy = Node Energy (for initial jump)
            path_energy = e_node
            
            # Construct node object
            cost = self._get_cost(nid)
            c = self.gravity.concepts[nid]
            
            r_node = RetrievalNode(
                id=nid,
                mass=c.mass,
                sim_to_query=sim,
                cost_total=cost.total,
                entropy=cost.entropy,
                instability=cost.instability,
                resistance=cost.resistance,
                energy_from_query=e_node,
                path_energy=path_energy,
                parent_id="QUERY",
                depth=1
            )
            
            heapq.heappush(pq, (path_energy, nid))
            visited[nid] = path_energy
            found_nodes[nid] = r_node

        # 3. Expand
        final_selection: Dict[str, RetrievalNode] = {}
        
        while pq and len(final_selection) < final_k:
            p_energy, curr_id = heapq.heappop(pq)
            
            # If we found a cheaper path to this node already, skip (lazy specific)
            if curr_id in visited and visited[curr_id] < p_energy:
                continue
                
            # Commit this node to final selection
            curr_node = found_nodes[curr_id]
            final_selection[curr_id] = curr_node
            
            # Stop expanding if max depth reached
            if curr_node.depth >= max_depth:
                continue
                
            # Expand neighbors
            # 1. Graph neighbors
            neighbors = []
            for (n1, n2), strength in self.gravity.relations.items():
                if strength > 0.1:
                    other = None
                    if n1 == curr_id: other = n2
                    elif n2 == curr_id: other = n1
                    if other: neighbors.append(other)
            
            # 2. Add geometric neighbors if graph is sparse?
            # (Optional, skipping for speed/strictness per design doc "fallback" exists)
            
            for next_id in neighbors:
                if next_id not in self.gravity.concepts: continue
                # Skip if cycle back to parent (simple check)
                if next_id == curr_node.parent_id: continue
                
                cost_next = self._get_cost(next_id)
                e_edge, sim_edge, rel_str = self._edge_energy(curr_id, next_id, cost_next)
                
                new_path_energy = p_energy + e_edge
                
                # Check if this is a better path or new node
                if next_id not in visited or new_path_energy < visited[next_id]:
                    visited[next_id] = new_path_energy
                    
                    # Calculate node-query properties for the record
                    sim_q_next = cosine(query_vec, self.gravity.concepts[next_id].vec)
                    e_node_next = self._node_energy(next_id, query_vec, sim_q_next)
                    c_next = self.gravity.concepts[next_id]
                    
                    next_node_obj = RetrievalNode(
                        id=next_id,
                        mass=c_next.mass,
                        sim_to_query=sim_q_next,
                        cost_total=cost_next.total,
                        entropy=cost_next.entropy,
                        instability=cost_next.instability,
                        resistance=cost_next.resistance,
                        energy_from_query=e_node_next,
                        path_energy=new_path_energy,
                        parent_id=curr_id,
                        depth=curr_node.depth + 1
                    )
                    
                    found_nodes[next_id] = next_node_obj
                    heapq.heappush(pq, (new_path_energy, next_id))

        # 4. Reconstruct Edges for the Final Tree
        # The 'parent_id' in RetrievalNode defines the Minimum Energy Tree structure implicitly.
        final_edges = []
        
        for nid, node in final_selection.items():
            if node.parent_id and node.parent_id != "QUERY":
                if node.parent_id in final_selection:
                    # Re-compute edge details
                    cost_v = self._get_cost(nid)
                    e_edge, sim, rel = self._edge_energy(node.parent_id, nid, cost_v)
                    
                    edge = RetrievalEdge(
                        source=node.parent_id,
                        target=nid,
                        similarity=sim,
                        relation_strength=rel,
                        edge_energy=e_edge
                    )
                    final_edges.append(edge)
        
        return RetrievalTree(
            root_id="QUERY",
            nodes=final_selection,
            edges=final_edges
        )
