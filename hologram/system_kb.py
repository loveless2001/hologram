"""
Core Hologram system concepts (Tier 2).
These concepts describe the Hologram architecture itself and act as fixed anchors.
"""

SYSTEM_CONCEPTS = """
# Hologram System Architecture

## Core Concepts

Concept Drift: The process where semantically similar concepts attract each other in vector space while dissimilar concepts repel.

Concept Fusion: Merging two highly similar concepts into a single canonical concept with combined mass and relations.

Concept Mitosis: Splitting a concept under semantic tension into two distinct concepts when neighbors form bimodal clusters.

Gravitational Field: The high-dimensional vector space where concepts exist and interact through semantic forces.

Mass: A concept's accumulated evidence and influence, which affects gravitational pull and fusion threshold calibration.

Glyph: A symbolic anchor representing a collection of related traces, acting as a massive attractor in the field.

Trace: A recorded piece of information (text, image, etc.) with an embedded vector representation.

Memory Packet: A structured subgraph extraction containing nodes, edges, glyphs, and trajectory information.

Probe: A semantic query vector that drifts through the gravitational field to discover relevant memory regions.

## Physics Parameters

Fusion Threshold: Base similarity threshold (0.85) required for concept fusion, adjusted by mass (black hole effect).

Mitosis Threshold: Minimum centroid separation (0.3) required for concept splitting.

Mass Decay: Rate at which unreinforced concepts lose mass over time (0.95 per step).

Isolation Drift: Rate at which unreinforced concepts drift away from the field centroid (0.01).

Quantization Level: Minimum action threshold below which drift operations are suppressed (calibrated by hardware).

Gamma Decay: Rate at which relation strengths decay over time (0.98 per step).

## 3-Tier Ontology

Tier 1 Domain Concepts: Dynamic memory from knowledge bases and user input, fully participates in physics.

Tier 2 System Concepts: Hologram architecture descriptions, act as fixed anchors with no physics.

Tier 3 Meta-Operators: The fundamental laws themselves, exist outside vector space.

Cross-Domain Isolation: Concepts from different projects cannot fuse, preventing contamination.

Origin Type: Classifies concept source as kb, runtime, manual, or system_design for validation.

Namespace Protection: Concepts with system: prefix are protected from fusion and mitosis.

## Dynamic Regulation

Step Dynamics: Orchestrates automatic fusion and mitosis during each concept addition.

Black Hole Effect: Massive concepts lower their fusion threshold to capture nearby concepts.

Bimodal Separation: Mitosis criterion requiring distinct neighbor clusters via k-means.

Cooldown Period: Minimum steps between fusion/mitosis events to prevent oscillation.

Neighborhood Divergence: Jaccard distance check to prevent fusing concepts with divergent relations.
"""

def get_system_concepts():
    """Returns the system KB content."""
    return SYSTEM_CONCEPTS
