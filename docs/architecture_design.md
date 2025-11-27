# Hologram: Architectural Design

> **Version**: 1.0.0
> **Date**: 2025-11-27
> **Status**: Active Research

## 1. System Philosophy: "Memory as Gravity"

Traditional knowledge graphs treat memory as static storage: nodes are created, edges are explicitly defined, and retrieval is a lookup process. **Hologram** fundamentally rejects this model.

In Hologram, **memory is a gravitational field**.
- **Concepts** are massive bodies that warp the semantic space around them.
- **Relations** are not explicit links, but the result of gravitational attraction between concepts.
- **Retrieval** is not a search, but a simulation: we drop a "probe" (query) into the field and see where it drifts.
- **Structure** is emergent. It forms dynamically as concepts clump together (clustering) or split apart (mitosis) under the pressure of new information.

This approach allows the system to handle ambiguity, contradiction, and evolution naturally, mirroring the fluid nature of human memory.

## 2. Architectural Layers

The system is composed of three distinct but interacting layers, moving from raw data to symbolic structure.

### Layer 1: The Manifold (Encoding)
**Goal**: Unify all inputs (text, images, search queries) into a single, consistent, normalized vector space.

- **Responsibility**:
    - Encodes raw data into high-dimensional vectors.
    - Projects vectors onto a unit hypersphere (L2 normalization).
    - Aligns different modalities (e.g., CLIP for images, GLiNER/SentenceTransformers for text) into a shared latent space.
- **Key Component**: `LatentManifold` (`hologram/manifold.py`)
    - Acts as the gatekeeper. No vector enters the system without passing through the Manifold.

### Layer 2: The Field (Physics)
**Goal**: Simulate the semantic interactions between concepts.

- **Responsibility**:
    - Manages a collection of "Concepts" (nodes with vector + mass).
    - Calculates gravitational forces, forces are computed over the cosine geometry of the manifold (e.g., similarity and distance).
    - Updates positions: Concepts drift towards similar neighbors.
    - **Mitosis**: Detects when a concept is "pulled apart" by opposing meanings (bimodal distribution of neighbors) and splits it into two distinct variants (e.g., "bank_finance" vs "bank_river").
- **Key Component**: `GravityField` / `Gravity` (`hologram/gravity.py`)
    - The physics engine.
    - Uses **FAISS K-means** for geometry-based mitosis detection.

### Layer 3: The Graph (Symbolic)
**Goal**: Provide stable, human-readable anchors for the fluid physics layer.

- **Responsibility**:
    - Manages **Glyphs**: Permanent symbolic identifiers (e.g., "User", "Project X", "🝞").
    - Glyphs are not static labels; they are **massive objects** in the Gravity Field.
    - **Glyph Physics**:
        - **Vector**: Centroid of all traces attached to the glyph.
        - **Mass**: Logarithmic growth based on trace count ($M = 1 + 0.75 \ln(1 + N)$).
        - This ensures that important glyphs exert strong gravity, organizing related concepts around them.
- **Key Component**: `GlyphRegistry` (`hologram/glyphs.py`)

## 3. Module Design & Rationale

### `hologram/manifold.py`
**Rationale**: Early versions suffered from "vector drift" where text and image vectors lived in different scales.
**Design**:
- `LatentManifold` class wraps encoders.
- Enforces `v /= norm(v)` on every input.
- Provides a single `align_text` / `align_image` API.

### `hologram/gravity.py`
**Rationale**: We needed a way to make memory "active".
**Design**:
- `Concept` dataclass: stores `vec`, `mass`, `count`.
- `add_concept`:
    - **Reinforcement**: If concept exists, increase mass (memory consolidation).
    - **Mitosis**: If concept is new/modified, check if it bridges two distant clusters.
- `check_mitosis`:
    - **Old**: Heuristic threshold.
    - **New**: **Geometry-Based**. Uses `faiss.Kmeans(k=2)` to cluster neighbors. If centroids are far apart, split the concept.

### `hologram/glyphs.py`
**Rationale**: Pure vector fields are hard to navigate. We need named anchors.
**Design**:
- `Glyph`: A named entity that aggregates `Traces`.
- `attach_trace`:
    1.  Links trace to glyph.
    2.  **Recomputes Centroid**: Glyph vector = mean(trace vectors).
    3.  **Updates Gravity**: Pushes `glyph:{id}` to `GravityField` with updated mass/vector.

### `hologram/api.py`
**Rationale**: A unified facade for the complex subsystems.
**Design**:
- `Hologram` class initializes Manifold, Gravity, and GlyphRegistry.
- `add_text`:
    1.  Encode via Manifold.
    2.  Create Trace -> Attach to Glyph.
    3.  Extract sub-concepts -> Add to Gravity.
    4.  Trigger Mitosis checks.

## 4. Data Flow

### Ingestion Flow
1.  **User Input**: `add_text("Project Alpha", "The deadline is tight.")`
2.  **Manifold**: Encodes "The deadline is tight" -> `vec_trace`.
3.  **Glyph**:
    - Attaches trace to "Project Alpha".
    - Recalculates "Project Alpha" centroid & mass.
    - Pushes "glyph:Project Alpha" to Gravity.
4.  **Extraction**: GLiNER extracts "deadline".
5.  **Gravity**:
    - Adds "deadline" concept (vector + mass).
    - Links "deadline" <-> "glyph:Project Alpha" (attraction).
    - Checks "deadline" for mitosis (is "deadline" ambiguous?).

### Retrieval Flow
1.  **User Query**: `search("When is it due?")`
2.  **Manifold**: Encodes query -> `vec_query`.
3.  **Gravity**:
    - Spawns a "Probe" at `vec_query`.
    - Simulates physics: Probe drifts towards heavy, relevant concepts (e.g., "deadline", "Project Alpha").
4.  **Subgraph**: Returns the concepts and glyphs that pulled the probe the hardest.

## 5. LLM Integration & Symbolic Memory Interface (SMI)

### **Purpose**

The LLM is not the memory itself.
It is the **interpreter** of the holographic field.

While the Gravity Field organizes concepts and glyphs into a continuously evolving latent manifold, the LLM reconstructs coherent explanations, narratives, or answers by *reading* the structure of the field in symbolic form.

This is the function of the **SMI — Symbolic Memory Interface**.

### 5.1 SMI Overview: “Field → Symbols → Language”

The SMI provides a bridge between two worlds:

| Layer     | Representation                    | Behavior                       |
| --------- | --------------------------------- | ------------------------------ |
| **Field** | Vectors, masses, relations, drift | Physics (continuous evolution) |
| **Graph** | Concepts, glyphs, adjacency       | Structured symbolic state      |
| **LLM**   | Natural language                  | Coherent reconstruction        |

The SMI acts as a **translator**:

1. Encodes the current state of the Field into a symbolic graph snapshot.
2. Feeds this snapshot into an LLM.
3. The LLM synthesizes meaning from the structure — not from hidden weights.

In other words:

> **The LLM is querying memory, not hallucinating it.**

### 5.2 Subgraph Extraction (Memory Packet)

Given a query, the system constructs a *Memory Packet* — a JSON-serializable structure representing the most relevant region of the semantic field.

### **Components of a Memory Packet**

```json
{
    "seed": "speed of light",
    "nodes": [
        {"name": "time dilation", "mass": 1.3, "vector": [...], "age": 12},
        {"name": "speed of light", "mass": 1.5, "vector": [...], "age": 4}
    ],
    "edges": [
        {"a": "speed of light", "b": "time dilation", "relation": 0.82, "tension": 0.12}
    ],
    "glyphs": [
        {"id": "🝞", "mass": 2.4}
    ]
}
```

This packet is:

* **compact** (top-k concepts)
* **symbolic** (human-legible node names)
* **structured** (edges with relation/tension)
* **physics-informed** (masses, drift age, decay)

It captures **what the field remembers** at that moment.

### 5.3 LLM Prompting Protocol

The LLM receives a structured memory packet and a task instruction.

### **Minimal prompt template**

```text
You are reading a slice of holographic memory. 
Below is a structured semantic field extracted near the user query.

Nodes contain: name, mass (importance), and age (recency).
Edges contain: relation strength and semantic tension.
Glyphs represent stable, symbolic anchors.

Using ONLY the information in this memory field, reconstruct an explanation.
Avoid adding external facts. Missing details should be inferred from structural patterns.
```

### **LLM Input**

* Query text
* Memory Packet (JSON)

### **LLM Output**

* Reconstruction
* Explanation
* Multi-step reasoning grounded in the structure

This transforms the LLM into a **reader** of memory, not the memory itself.

### 5.4 Why the SMI Matters

#### **1. Prevents Hallucinations**

The LLM is constrained to:

* the graph
* the relations
* the mass distribution
* the tension between nodes

It cannot invent isolated facts; it must reason from structure.

#### **2. Enables Symbolic Recurrence**

Glyphs act as timeless anchors.
The LLM reactivates them even if:

* wording changes
* exact embeddings drift
* context evolves

This is how **semantic continuity** emerges without persistent text logs.

#### **3. Supports Multi-Session Memory**

Because the field is dynamic:

* high-mass nodes become “core memory”
* low-mass nodes decay
* mitosis resolves ambiguity over time
* glyphs maintain long-term identity

The SMI lets the LLM interpret this evolution.

#### **4. Mirrors Human Memory Use**

Humans do not retrieve verbatim logs — we reconstruct from:

* clusters
* associations
* weights
* symbolic cues

The SMI replicates this reconstruction mechanism.

### 5.5 SMI as a Research Layer

At this stage, the SMI is **not a rules engine or knowledge graph**.
It is:

* an interface
* a forcing function
* a binding layer

Future directions (for v1.1+) include:

* resonance-based packet selection
* probe trajectory visualization
* iterative multi-hop field reading
* LLM-generated “memory edits” (e.g., causally modifying the field)
* symbolic compression for long-term storage

But the core principle remains:

> **The SMI lets the LLM access memory without contaminating the field with its own internal hallucinations.**

## 6. Version History

| Version | Date | Changes |
| :--- | :--- | :--- |
| **0.1.0** | 2025-10-01 | Initial Prototype. Simple vector storage. |
| **0.5.0** | 2025-11-15 | Added Gravity Field simulation. |
| **0.8.0** | 2025-11-20 | Added Concept Mitosis (Heuristic). |
| **1.0.0** | 2025-11-27 | **Research Upgrade**: LatentManifold, Geometry-Based Mitosis (FAISS), Glyph Physics, SMI. |
