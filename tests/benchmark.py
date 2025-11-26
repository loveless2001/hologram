#!/usr/bin/env python3
"""
Benchmark Suite for Holographic Memory

Implements the metrics defined in docs/performance_metrics.md
"""

import numpy as np
import time
from typing import List, Tuple, Dict
from hologram.api import Hologram

class HologramBenchmark:
    """Benchmark suite for evaluating holographic memory performance."""
    
    def __init__(self, hg: Hologram):
        self.hg = hg
        self.results = {}
    
    # ===== 1. RETRIEVAL QUALITY =====
    
    def precision_at_k(self, queries: List[Tuple[str, List[str]]], k: int = 10) -> float:
        """
        Calculate Precision@K for a set of queries.
        
        Args:
            queries: List of (query_text, list_of_relevant_ids)
            k: Number of top results to consider
            
        Returns:
            Average precision across all queries
        """
        precisions = []
        
        for query_text, relevant_ids in queries:
            results = self.hg.search_text(query_text, top_k=k)
            retrieved_ids = [trace.trace_id for trace, _ in results]
            
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            precision = relevant_retrieved / k
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def recall_at_k(self, queries: List[Tuple[str, List[str]]], k: int = 10) -> float:
        """Calculate Recall@K."""
        recalls = []
        
        for query_text, relevant_ids in queries:
            if not relevant_ids:
                continue
                
            results = self.hg.search_text(query_text, top_k=k)
            retrieved_ids = [trace.trace_id for trace, _ in results]
            
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            recall = relevant_retrieved / len(relevant_ids)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def mean_reciprocal_rank(self, queries: List[Tuple[str, List[str]]]) -> float:
        """Calculate MRR (Mean Reciprocal Rank)."""
        reciprocal_ranks = []
        
        for query_text, relevant_ids in queries:
            results = self.hg.search_text(query_text, top_k=20)
            
            for i, (trace, _) in enumerate(results, 1):
                if trace.trace_id in relevant_ids:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)  # No relevant result found
        
        return np.mean(reciprocal_ranks)
    
    # ===== 2. GRAVITY FIELD QUALITY =====
    
    def attraction_score(self, positive_triplets: List[Tuple[str, str, str]]) -> float:
        """
        Measure attraction between concepts after positive statements.
        
        Args:
            positive_triplets: List of (concept_a, relation, concept_b)
            
        Returns:
            Average ratio of similarity after/before (> 1.0 = attraction)
        """
        ratios = []
        
        for concept_a, relation, concept_b in positive_triplets:
            # Add concepts
            id_a = self.hg.add_text("test", concept_a, trace_id=f"test_a_{len(ratios)}")
            id_b = self.hg.add_text("test", concept_b, trace_id=f"test_b_{len(ratios)}")
            
            # Measure before
            vec_a = self.hg.store.sim.concepts[id_a].vec.copy()
            vec_b = self.hg.store.sim.concepts[id_b].vec.copy()
            sim_before = np.dot(vec_a, vec_b)
            
            # Add linking statement
            self.hg.add_text("test", f"{concept_a} {relation} {concept_b}")
            
            # Measure after
            vec_a_after = self.hg.store.sim.concepts[id_a].vec
            vec_b_after = self.hg.store.sim.concepts[id_b].vec
            sim_after = np.dot(vec_a_after, vec_b_after)
            
            if sim_before != 0:
                ratios.append(sim_after / sim_before)
        
        return np.mean(ratios) if ratios else 0.0
    
    def repulsion_score(self, negative_triplets: List[Tuple[str, str, str]]) -> float:
        """
        Measure repulsion between concepts after negative statements.
        
        Returns:
            Average ratio of similarity after/before (< 1.0 = repulsion)
        """
        ratios = []
        
        for concept_a, relation, concept_b in negative_triplets:
            # Add concepts
            id_a = self.hg.add_text("test", concept_a, trace_id=f"test_neg_a_{len(ratios)}")
            id_b = self.hg.add_text("test", concept_b, trace_id=f"test_neg_b_{len(ratios)}")
            
            # Measure before
            vec_a = self.hg.store.sim.concepts[id_a].vec.copy()
            vec_b = self.hg.store.sim.concepts[id_b].vec.copy()
            sim_before = np.dot(vec_a, vec_b)
            
            # Add negated statement
            self.hg.add_text("test", f"{concept_a} is NOT {relation} {concept_b}")
            
            # Measure after
            vec_a_after = self.hg.store.sim.concepts[id_a].vec
            vec_b_after = self.hg.store.sim.concepts[id_b].vec
            sim_after = np.dot(vec_a_after, vec_b_after)
            
            if sim_before != 0:
                ratios.append(sim_after / sim_before)
        
        return np.mean(ratios) if ratios else 0.0
    
    # ===== 3. TEMPORAL DYNAMICS =====
    
    def measure_decay(self, n_steps: int = 50) -> Dict[str, float]:
        """
        Measure decay behavior over time.
        
        Returns:
            Dict with 'half_life', 'final_mass_ratio', etc.
        """
        # Create test concept
        test_id = self.hg.add_text("decay_test", "Test concept for decay")
        initial_mass = self.hg.store.sim.concepts[test_id].mass
        
        masses = [initial_mass]
        
        # Let it decay without reinforcement
        for step in range(n_steps):
            # Add other concepts to advance global_step
            self.hg.add_text("noise", f"Noise {step}")
            self.hg.decay(steps=1)
            
            current_mass = self.hg.store.sim.concepts[test_id].mass
            masses.append(current_mass)
        
        # Calculate half-life (steps to reach 50% mass)
        half_mass = initial_mass * 0.5
        half_life = None
        for i, mass in enumerate(masses):
            if mass <= half_mass:
                half_life = i
                break
        
        final_ratio = masses[-1] / initial_mass
        
        return {
            'half_life': half_life,
            'initial_mass': initial_mass,
            'final_mass': masses[-1],
            'final_ratio': final_ratio,
            'mass_history': masses
        }
    
    def measure_reinforcement(self, n_steps: int = 50, reinforce_every: int = 5) -> Dict[str, float]:
        """
        Measure reinforcement effectiveness.
        
        Returns:
            Dict with mass gain from reinforcement
        """
        # Create two concepts: one reinforced, one not
        freq_id = self.hg.add_text("reinforce_test", "Frequent concept")
        rare_id = self.hg.add_text("reinforce_test", "Rare concept")
        
        initial_mass_freq = self.hg.store.sim.concepts[freq_id].mass
        initial_mass_rare = self.hg.store.sim.concepts[rare_id].mass
        
        for step in range(n_steps):
            if step % reinforce_every == 0:
                self.hg.add_text("reinforce_test", "Frequent concept")
            
            self.hg.add_text("noise", f"Noise {step}")
            self.hg.decay(steps=1)
        
        final_mass_freq = self.hg.store.sim.concepts[freq_id].mass
        final_mass_rare = self.hg.store.sim.concepts[rare_id].mass
        
        return {
            'reinforced_gain': final_mass_freq / initial_mass_freq,
            'unreinforced_ratio': final_mass_rare / initial_mass_rare,
            'final_mass_frequent': final_mass_freq,
            'final_mass_rare': final_mass_rare
        }
    
    # ===== 4. EFFICIENCY =====
    
    def measure_query_latency(self, queries: List[str], n_runs: int = 10) -> Dict[str, float]:
        """Measure average query latency."""
        latencies = []
        
        for _ in range(n_runs):
            for query in queries:
                start = time.time()
                self.hg.search_text(query, top_k=10)
                end = time.time()
                latencies.append((end - start) * 1000)  # ms
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
    
    def measure_indexing_throughput(self, n_concepts: int = 100) -> float:
        """Measure concepts indexed per second."""
        concepts = [f"Concept {i}" for i in range(n_concepts)]
        
        start = time.time()
        for concept in concepts:
            self.hg.add_text("throughput_test", concept)
        end = time.time()
        
        duration = end - start
        return n_concepts / duration  # concepts/sec
    
    # ===== 5. RUN ALL BENCHMARKS =====
    
    def run_full_suite(self, test_data: Dict) -> Dict:
        """
        Run complete benchmark suite.
        
        Args:
            test_data: Dict containing test queries, triplets, etc.
            
        Returns:
            Dict of all benchmark results
        """
        print("Running Holographic Memory Benchmark Suite...")
        print("=" * 60)
        
        results = {}
        
        # Retrieval Quality
        if 'retrieval_queries' in test_data:
            print("\n[1/5] Measuring Retrieval Quality...")
            queries = test_data['retrieval_queries']
            results['precision@10'] = self.precision_at_k(queries, k=10)
            results['recall@10'] = self.recall_at_k(queries, k=10)
            results['MRR'] = self.mean_reciprocal_rank(queries)
            print(f"  Precision@10: {results['precision@10']:.3f}")
            print(f"  Recall@10:    {results['recall@10']:.3f}")
            print(f"  MRR:          {results['MRR']:.3f}")
        
        # Gravity Field
        if 'positive_triplets' in test_data and 'negative_triplets' in test_data:
            print("\n[2/5] Measuring Gravity Field Behavior...")
            results['attraction_score'] = self.attraction_score(test_data['positive_triplets'])
            results['repulsion_score'] = self.repulsion_score(test_data['negative_triplets'])
            print(f"  Attraction:   {results['attraction_score']:.3f} (expect > 1.0)")
            print(f"  Repulsion:    {results['repulsion_score']:.3f} (expect < 1.0)")
        
        # Temporal Dynamics
        print("\n[3/5] Measuring Temporal Dynamics...")
        decay_results = self.measure_decay(n_steps=50)
        reinforce_results = self.measure_reinforcement(n_steps=50, reinforce_every=5)
        results.update({f'decay_{k}': v for k, v in decay_results.items()})
        results.update({f'reinforce_{k}': v for k, v in reinforce_results.items()})
        print(f"  Decay half-life:      {decay_results['half_life']} steps")
        print(f"  Reinforcement gain:   {reinforce_results['reinforced_gain']:.2f}x")
        
        # Efficiency
        if 'performance_queries' in test_data:
            print("\n[4/5] Measuring Efficiency...")
            latency = self.measure_query_latency(test_data['performance_queries'])
            throughput = self.measure_indexing_throughput(n_concepts=100)
            results['query_latency'] = latency
            results['indexing_throughput'] = throughput
            print(f"  Avg query latency:    {latency['mean_latency_ms']:.2f} ms")
            print(f"  Indexing throughput:  {throughput:.1f} concepts/sec")
        
        print("\n[5/5] Benchmark Complete!")
        print("=" * 60)
        
        self.results = results
        return results
    
    def export_results(self, filepath: str):
        """Export results to JSON."""
        import json
        with open(filepath, 'w') as f:
            # Convert numpy types for JSON serialization
            clean_results = {}
            for k, v in self.results.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_results[k] = float(v)
                elif isinstance(v, dict):
                    clean_results[k] = {
                        k2: float(v2) if isinstance(v2, (np.integer, np.floating)) else v2
                        for k2, v2 in v.items()
                        if not isinstance(v2, (list, np.ndarray))  # Skip arrays
                    }
                else:
                    clean_results[k] = v
            
            json.dump(clean_results, f, indent=2)
        print(f"\nResults exported to: {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize system
    hg = Hologram.init(use_clip=False)
    hg.glyphs.create("benchmark", title="Benchmark Test")
    
    # Prepare test data
    test_data = {
        'retrieval_queries': [
            # (query, [relevant_ids])
            ("gravity", []),  # Will be populated after adding data
        ],
        'positive_triplets': [
            ("Mass", "increases with", "Velocity"),
            ("Energy", "equals", "Mass"),
        ],
        'negative_triplets': [
            ("Mass", "independent of", "Color"),
        ],
        'performance_queries': [
            "what is gravity",
            "how does mass behave",
        ]
    }
    
    # Add some test data
    test_concepts = [
        "Gravity attracts masses",
        "Mass is a property of matter",
        "Velocity measures speed",
        "Energy is conserved",
    ]
    
    for concept in test_concepts:
        hg.add_text("benchmark", concept)
    
    # Run benchmark
    benchmark = HologramBenchmark(hg)
    results = benchmark.run_full_suite(test_data)
    benchmark.export_results("benchmark_results.json")
