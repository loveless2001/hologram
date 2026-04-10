#!/usr/bin/env python3
"""
Benchmark: Global vs Glyph-Routed Retrieval

Compares retrieval quality between global (search_across) and glyph-routed
(search_routed) paths. Control condition: raw cosine only, no probe reranking
or mass-weighted rerank on either path.

Metrics: Recall@5, Interference Rate, Latency
"""
import os
os.environ["HOLOGRAM_QUIET"] = "1"

import time
import numpy as np
from hologram.store import MemoryStore, Trace
from hologram.glyphs import GlyphRegistry
from hologram.glyph_router import GlyphRouter

# --- Configuration ---
DIM = 128
TRACES_PER_DOMAIN = 15
TOP_K = 5
RUNS = 20  # queries per domain
BIAS_STRENGTH = 1.5  # Lower = more domain overlap = harder for global

# --- Domain definitions ---
# Each domain gets a distinct bias region in embedding space
DOMAINS = {
    "physics": {"bias_start": 0, "bias_end": 32, "terms": [
        "relativity", "quantum mechanics", "gravity", "spacetime curvature",
        "photon", "wave function", "black hole", "entropy",
        "electromagnetism", "higgs boson", "dark matter", "cosmic inflation",
        "particle accelerator", "string theory", "quantum entanglement"
    ]},
    "biology": {"bias_start": 32, "bias_end": 64, "terms": [
        "dna replication", "cell membrane", "protein folding", "mitosis",
        "rna transcription", "enzyme catalysis", "photosynthesis", "neuron",
        "immune response", "genetic mutation", "evolution", "chromosome",
        "metabolism", "stem cell", "ribosome"
    ]},
    "computing": {"bias_start": 64, "bias_end": 96, "terms": [
        "hash table", "binary search", "neural network", "compiler",
        "operating system", "tcp protocol", "encryption", "database index",
        "garbage collection", "distributed system", "cache invalidation",
        "recursion", "api gateway", "container orchestration", "load balancer"
    ]},
}


def make_domain_vec(dim, bias_start, bias_end, strength=None):
    """Create a random vector biased toward a specific dimension region."""
    vec = np.random.randn(dim).astype("float32")
    vec[bias_start:bias_end] += BIAS_STRENGTH if strength is None else strength
    return vec


def setup_kb():
    """Create a multi-domain KB with 3 glyphs and domain-separated traces."""
    store = MemoryStore(vec_dim=DIM)
    glyphs = GlyphRegistry(store)

    ground_truth = {}  # query_domain -> set of trace_ids that are relevant

    for domain, cfg in DOMAINS.items():
        glyphs.create(domain, title=domain.capitalize())
        domain_trace_ids = set()

        for i, term in enumerate(cfg["terms"][:TRACES_PER_DOMAIN]):
            vec = make_domain_vec(DIM, cfg["bias_start"], cfg["bias_end"])
            tid = f"{domain}_{i}"
            t = Trace(trace_id=tid, kind="text", content=term, vec=vec)
            glyphs.attach_trace(domain, t)
            domain_trace_ids.add(tid)

        ground_truth[domain] = domain_trace_ids

    router = GlyphRouter(store, glyphs)
    return store, glyphs, router, ground_truth


def run_benchmark():
    """Run comparative benchmark."""
    np.random.seed(42)
    store, glyphs, router, ground_truth = setup_kb()

    results = {"global": {}, "routed": {}}

    for method_name in ["global", "routed"]:
        total_recall = 0
        total_interference = 0
        total_queries = 0
        latencies = []

        for domain, cfg in DOMAINS.items():
            for _ in range(RUNS):
                # Generate domain-biased query
                q_vec = make_domain_vec(DIM, cfg["bias_start"], cfg["bias_end"])

                # Time the search
                t0 = time.perf_counter()
                if method_name == "global":
                    hits = store.search_traces(q_vec, top_k=TOP_K)
                    result_ids = [tid for tid, _ in hits]
                else:
                    # Realistic config: fallback enabled, routing filters+promotes
                    hits = router.search_routed(q_vec, top_k=TOP_K,
                                                 fallback_global=True)
                    result_ids = [tid for tid, _ in hits]
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)

                # Compute metrics
                relevant = ground_truth[domain]
                retrieved = set(result_ids[:TOP_K])

                recall = len(retrieved & relevant) / min(TOP_K, len(relevant))
                interference = len(retrieved - relevant) / max(1, len(retrieved))

                total_recall += recall
                total_interference += interference
                total_queries += 1

        results[method_name] = {
            "avg_recall_at_5": total_recall / total_queries,
            "avg_interference_rate": total_interference / total_queries,
            "avg_latency_ms": np.mean(latencies) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "total_queries": total_queries,
        }

    return results


def print_results(results):
    """Print formatted benchmark results."""
    print("=" * 60)
    print("BENCHMARK: Global vs Glyph-Routed Retrieval")
    print("=" * 60)
    print(f"Config: DIM={DIM}, TRACES_PER_DOMAIN={TRACES_PER_DOMAIN}, "
          f"TOP_K={TOP_K}, RUNS={RUNS}")
    print(f"Domains: {list(DOMAINS.keys())}")
    print()

    header = f"{'Metric':<25} {'Global':>12} {'Routed':>12} {'Delta':>12}"
    print(header)
    print("-" * 61)

    g = results["global"]
    r = results["routed"]

    metrics = [
        ("Recall@5", "avg_recall_at_5", True),
        ("Interference Rate", "avg_interference_rate", False),
        ("Avg Latency (ms)", "avg_latency_ms", False),
        ("P95 Latency (ms)", "p95_latency_ms", False),
    ]

    for label, key, higher_better in metrics:
        gv = g[key]
        rv = r[key]
        delta = rv - gv
        sign = "+" if delta > 0 else ""
        marker = ""
        if key in ("avg_recall_at_5", "avg_interference_rate"):
            if higher_better and delta > 0:
                marker = " <-- better"
            elif not higher_better and delta < 0:
                marker = " <-- better"
        print(f"{label:<25} {gv:>12.4f} {rv:>12.4f} {sign}{delta:>11.4f}{marker}")

    print()

    # Verdict
    recall_ok = r["avg_recall_at_5"] >= g["avg_recall_at_5"] - 0.05
    interference_better = r["avg_interference_rate"] < g["avg_interference_rate"]

    if recall_ok and interference_better:
        print("VERDICT: PASS — Routed retrieval reduces interference without "
              "recall regression.")
    elif recall_ok and not interference_better:
        print("VERDICT: NEUTRAL — No recall regression, but interference not "
              "reduced. Routing may not add value yet.")
    else:
        print("VERDICT: FAIL — Recall regression detected. Investigate glyph "
              "inference quality.")


if __name__ == "__main__":
    results = run_benchmark()
    print_results(results)
