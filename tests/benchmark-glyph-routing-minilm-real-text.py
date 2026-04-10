#!/usr/bin/env python3
"""
Benchmark: Global vs Glyph-Routed Retrieval with MiniLM Real Text Embeddings

Tests whether random R_g + P_k transforms help with real semantic embeddings
where domain signal is distributed (not axis-aligned). Decision gate for
whether learned R_g is needed.

Compares: global search vs routed (identity) vs routed (R_g + P_k)

Important latency note:
- Routed search lazily builds glyph operators and shard indexes.
- For projected methods, the first timed routed query includes cold-path work
  such as QR decomposition, trace transforms, and FAISS shard construction.
- Quality comparisons are still valid, but raw latency output should be read as
  cold-or-partially-amortized latency unless warm-up timing is added.
"""
import os
os.environ["HOLOGRAM_QUIET"] = "1"

import time
import numpy as np
from hologram.store import MemoryStore, Trace
from hologram.glyphs import GlyphRegistry
from hologram.glyph_router import GlyphRouter
from hologram.embeddings import TextMiniLM

# --- Domain text data ---
DOMAINS = {
    "physics": [
        "Einstein's theory of general relativity describes gravity as spacetime curvature",
        "Quantum entanglement allows particles to be correlated across large distances",
        "The Higgs boson gives mass to fundamental particles via the Higgs field",
        "Black holes form when massive stars collapse under their own gravity",
        "The speed of light in vacuum is approximately 299792458 meters per second",
        "Electromagnetic waves propagate through space at the speed of light",
        "Nuclear fusion powers the sun by converting hydrogen into helium",
        "Dark matter makes up about 27 percent of the total mass energy of the universe",
        "Superconductivity occurs when materials conduct electricity with zero resistance",
        "The standard model describes three of the four fundamental forces of nature",
        "Photons are massless particles that carry the electromagnetic force",
        "String theory proposes that fundamental particles are vibrating strings",
        "Cosmic inflation explains the uniformity of the cosmic microwave background",
        "Neutron stars are incredibly dense remnants of massive stellar explosions",
        "Wave-particle duality means light behaves as both a wave and a particle",
    ],
    "biology": [
        "DNA replication is the process by which a cell copies its genetic material",
        "Mitochondria generate ATP through oxidative phosphorylation in cells",
        "CRISPR-Cas9 enables precise editing of genomic sequences in living organisms",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
        "Neurons transmit electrical signals through axons to communicate information",
        "The immune system uses antibodies to identify and neutralize foreign pathogens",
        "Meiosis produces gametes with half the chromosome number of parent cells",
        "Enzymes are biological catalysts that speed up chemical reactions in cells",
        "Evolution by natural selection drives adaptation in populations over generations",
        "Ribosomes translate messenger RNA into proteins using transfer RNA molecules",
        "Stem cells can differentiate into many specialized cell types in the body",
        "The cell membrane controls what enters and exits the cell selectively",
        "Genetic mutations can be caused by errors in DNA replication or repair",
        "Epigenetics studies changes in gene expression without altering DNA sequence",
        "Chloroplasts contain chlorophyll which absorbs light energy for photosynthesis",
    ],
    "computing": [
        "Hash tables provide average constant-time lookup by mapping keys to buckets",
        "Neural networks learn hierarchical representations through backpropagation",
        "TCP ensures reliable ordered delivery of data packets across networks",
        "Garbage collection automatically reclaims memory from unreachable objects",
        "Distributed consensus algorithms like Raft maintain consistency across replicas",
        "Public key cryptography enables secure communication without shared secrets",
        "Database indexes use B-trees to accelerate query lookups on large datasets",
        "Container orchestration platforms like Kubernetes manage application deployment",
        "Binary search achieves logarithmic time complexity on sorted collections",
        "Compilers transform high-level source code into machine executable instructions",
        "Load balancers distribute incoming network traffic across multiple servers",
        "Cache invalidation ensures stale data is removed when the source changes",
        "Recursion solves problems by breaking them into smaller subproblems",
        "API gateways handle authentication routing and rate limiting for microservices",
        "Operating systems manage hardware resources and provide abstractions for programs",
    ],
}

# Test queries (clearly in-domain)
QUERIES = {
    "physics": [
        "How does gravity bend spacetime?",
        "What is quantum superposition?",
        "Explain nuclear fusion in stars",
        "Properties of electromagnetic radiation",
        "What is dark energy in cosmology?",
    ],
    "biology": [
        "How do cells divide during mitosis?",
        "What is the role of DNA in heredity?",
        "How do enzymes catalyze reactions?",
        "Explain the immune response to infection",
        "How does photosynthesis work in plants?",
    ],
    "computing": [
        "How do hash maps handle collisions?",
        "Explain backpropagation in neural networks",
        "How does TCP ensure reliable delivery?",
        "What is garbage collection in programming?",
        "How do database indexes improve performance?",
    ],
}

TOP_K = 5


def setup_kb(encoder):
    """Create KB with MiniLM-encoded real text."""
    dim = 384  # MiniLM dim
    store = MemoryStore(vec_dim=dim)
    glyphs = GlyphRegistry(store)
    ground_truth = {}

    for domain, texts in DOMAINS.items():
        glyphs.create(domain, title=domain.capitalize())
        domain_tids = set()

        for i, text in enumerate(texts):
            vec = encoder.encode(text)
            tid = f"{domain}_{i}"
            t = Trace(trace_id=tid, kind="text", content=text, vec=vec)
            glyphs.attach_trace(domain, t)
            domain_tids.add(tid)

        ground_truth[domain] = domain_tids

    return store, glyphs, ground_truth


def run_method(method_name, store, glyphs, encoder, ground_truth,
               use_projection=False, projection_k=None):
    """Run benchmark for one method."""
    router = GlyphRouter(store, glyphs, use_projection=use_projection,
                         projection_k=projection_k)

    total_recall = 0
    total_interference = 0
    total_queries = 0
    latencies = []

    for domain, queries in QUERIES.items():
        relevant = ground_truth[domain]
        for query in queries:
            q_vec = encoder.encode(query)

            t0 = time.perf_counter()
            if method_name == "global":
                hits = store.search_traces(q_vec, top_k=TOP_K)
                result_ids = [tid for tid, _ in hits]
            else:
                hits = router.search_routed(q_vec, top_k=TOP_K,
                                            fallback_global=True)
                result_ids = [tid for tid, _ in hits]
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            retrieved = set(result_ids[:TOP_K])
            recall = len(retrieved & relevant) / min(TOP_K, len(relevant))
            interference = len(retrieved - relevant) / max(1, len(retrieved))

            total_recall += recall
            total_interference += interference
            total_queries += 1

    return {
        "avg_recall_at_5": total_recall / total_queries,
        "avg_interference_rate": total_interference / total_queries,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "total_queries": total_queries,
    }


def main():
    print("Loading MiniLM encoder...")
    encoder = TextMiniLM()
    print("Encoding domain texts and building KB...")
    store, glyphs, ground_truth = setup_kb(encoder)

    configs = [
        ("Global (baseline)", "global", False, None),
        ("Routed (identity)", "routed", False, None),
        ("Routed (R_g+P_k k=48)", "routed", True, 48),
        ("Routed (R_g+P_k k=96)", "routed", True, 96),
        ("Routed (R_g+P_k k=192)", "routed", True, 192),
    ]

    results = {}
    for label, method, use_proj, k in configs:
        r = run_method(method, store, glyphs, encoder, ground_truth,
                       use_projection=use_proj, projection_k=k)
        results[label] = r

    # Print results
    print()
    print("=" * 72)
    print("BENCHMARK: Global vs Glyph-Routed (MiniLM Real Text Embeddings)")
    print("=" * 72)
    print(f"Encoder: MiniLM-L6-v2 (384d)")
    print(f"Domains: {list(DOMAINS.keys())}, {len(list(DOMAINS.values())[0])} traces each")
    print(f"Queries: {sum(len(q) for q in QUERIES.values())} total, TOP_K={TOP_K}")
    print()

    header = f"{'Method':<25} {'Recall@5':>10} {'Interf':>10} {'Lat(ms)':>10}"
    print(header)
    print("-" * 55)

    baseline = results["Global (baseline)"]
    for label, r in results.items():
        print(f"{label:<25} {r['avg_recall_at_5']:>10.4f} "
              f"{r['avg_interference_rate']:>10.4f} "
              f"{r['avg_latency_ms']:>10.2f}")

    print()
    # Verdict
    identity = results["Routed (identity)"]
    best_proj = min(
        [(l, r) for l, r in results.items() if "R_g" in l],
        key=lambda x: x[1]["avg_interference_rate"]
    )

    print(f"Best projection config: {best_proj[0]}")
    print(f"  vs Global: recall {best_proj[1]['avg_recall_at_5']:.4f} vs "
          f"{baseline['avg_recall_at_5']:.4f}, "
          f"interference {best_proj[1]['avg_interference_rate']:.4f} vs "
          f"{baseline['avg_interference_rate']:.4f}")

    if (best_proj[1]["avg_interference_rate"] < baseline["avg_interference_rate"]
            and best_proj[1]["avg_recall_at_5"] >= baseline["avg_recall_at_5"] - 0.05):
        print("\nVERDICT: PASS — R_g+P_k reduces interference on real embeddings.")
        print("Random rotations have value. Learned R_g is an optimization, not a requirement.")
    elif (identity["avg_interference_rate"] < baseline["avg_interference_rate"]
          and identity["avg_recall_at_5"] >= baseline["avg_recall_at_5"] - 0.05):
        print("\nVERDICT: ROUTING HELPS, PROJECTION NEUTRAL — Glyph routing alone "
              "reduces interference. R_g+P_k doesn't add value beyond routing.")
    else:
        print("\nVERDICT: INCONCLUSIVE — Neither routing nor projection clearly "
              "reduces interference. Consider learned R_g or different k values.")


if __name__ == "__main__":
    main()
