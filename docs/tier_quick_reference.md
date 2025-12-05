# 3-Tier Ontology Quick Reference

## Tier Levels

| Tier | Name | Description | Physics | Storage |
|------|------|-------------|---------|---------|
| **1** | Domain Concepts | User data, KB ingestion | ✅ Full (drift, fusion, mitosis) | Vector |
| **2** | System Concepts | Architecture, meta-knowledge | ❌ None (fixed anchors) | Vector |
| **3** | Meta-Operators | Laws, rules, constraints | ❌ None (not stored) | Rule-based |

## Key Functions

### Validation
```python
from hologram.gravity import can_interact, is_protected_namespace, TIER_DOMAIN, TIER_SYSTEM

# Check if two concepts can interact
can_interact(concept_a, concept_b)  
# Returns True only if:
# - Both are Tier 1
# - Same project
# - Same origin

# Check if name is protected
is_protected_namespace("system:gravity")  # True
is_protected_namespace("user:data")       # False
```

### Adding Concepts

```python
# Domain concept (Tier 1)
holo.add_text(
    "glyph_id",
    "text content",
    tier=1,              # TIER_DOMAIN
    origin="kb"          # "kb", "runtime", "manual"
)

# System concept (Tier 2) - usually auto-loaded
holo.field.sim.add_concept(
    "system:custom_concept",
    vec=custom_vector,
    tier=2,              # TIER_SYSTEM  
    project="hologram",
    origin="system_design"
)
```

### Project Namespaces

```python
# Set project for new concepts
holo.project = "ml_research"
holo.add_text("doc", "AI content", tier=1, origin="kb")

holo.project = "physics"  
holo.add_text("doc", "Physics content", tier=1, origin="kb")
# These won't fuse across projects
```

## Protected Namespaces

Concepts with these prefixes are immune to fusion/mitosis:
- `system:`
- `meta:`
- `hologram:`
- `architecture:`

## Origin Types

| Origin | Use Case | Protection |
|--------|----------|------------|
| `kb` | Knowledge base ingestion | Standard tier rules |
| `runtime` | Generated at runtime | Standard tier rules |
| `manual` | User-created | Standard tier rules |
| `system_design` | Architecture docs | **Mitosis blocked** |

## Concept Metadata

Every concept has:
```python
concept.tier               # 1, 2, or 3
concept.project            # "default", "ml_research", etc.
concept.origin             # "kb", "runtime", "manual", "system_design"
concept.last_fusion_step   # For cooldown tracking
concept.last_mitosis_step  # For cooldown tracking
```

## Physics Rules

### Fusion (Gravity)
```python
# Will fuse if ALL true:
✓ Both Tier 1 (Domain)
✓ Same project
✓ Same origin  
✓ Not in protected namespace
✓ Similarity > threshold (calibrated by mass)
✓ Passed cooldown period
✓ Neighborhoods not divergent (Jaccard > 0.4)
```

### Mitosis (Cell Division)
```python
# Will split if ALL true:
✓ Tier 1 (Domain)
✓ Mass ≥ threshold (default: 2.0)
✓ Not origin="system_design"
✓ Not in protected namespace
✓ Has ≥3 strong neighbors
✓ Neighbors form bimodal clusters
✓ Passed cooldown period
```

### Drift (Mutual Attraction/Repulsion)
```python
# Only applies to:
✓ Tier 1 (Domain) concepts
✗ Tier 2+ concepts are static
```

## Common Patterns

### Multi-Project Setup
```python
# Initialize
holo = Hologram.init(use_gravity=True, auto_ingest_system=True)

# Add concepts to different projects
for project_name, docs in project_docs.items():
    holo.project = project_name
    for doc in docs:
        holo.add_text(f"{project_name}_doc", doc, tier=1, origin="kb")
        
# Projects remain isolated
holo.field.sim.step_dynamics()  # No cross-project fusion
```

### System Concept Ingestion
```python
# Automatic (recommended)
holo = Hologram.init(auto_ingest_system=True)  # 31 system concepts loaded

# Manual
from hologram.system_kb import get_system_concepts
from hologram.text_utils import extract_concepts

sys_text = get_system_concepts()
concepts = extract_concepts(sys_text)
for c in concepts:
    vec = holo.manifold.align_text(c, holo.text_encoder)
    holo.field.add(f"system:{hash(c)}", vec, tier=2, project="hologram", origin="system_design")
```

### Save/Load
```python
# Save (defaults to SQLite 'memory.db' in project dir)
holo.save()

# Save to specific JSON file (legacy/backup)
holo.save("my_hologram.json")

# Load (auto-detects backend)
holo = Hologram.load("memory.db", auto_ingest_system=False)
```

## Debugging

### Check Concept Tier
```python
concept = holo.field.sim.concepts["concept_name"]
print(f"Tier: {concept.tier}")
print(f"Project: {concept.project}")
print(f"Origin: {concept.origin}")
```

### List by Tier
```python
tier1 = [c for c in holo.field.sim.concepts.values() if c.tier == 1]
tier2 = [c for c in holo.field.sim.concepts.values() if c.tier == 2]
print(f"Domain: {len(tier1)}, System: {len(tier2)}")
```

### Check Protection
```python
from hologram.gravity import is_protected_namespace

protected = [name for name in holo.field.sim.concepts 
             if is_protected_namespace(name)]
print(f"Protected concepts: {protected}")
```

## Verification Commands

```bash
# Run automated tests
pytest tests/test_tier_validation.py

# Run verification script
python3 scripts/verify_tier_system.py

# Generate visualization
python3 scripts/visualize_tiers.py

# Run full demo
python3 examples/tier_demo.py
```

## Performance Notes

- **Fusion check**: Filters concepts before FAISS search (O(N) filter + O(N log k) search)
- **Mitosis check**: Only top-10 massive concepts checked per step
- **Cooldown**: Prevents oscillation, improves performance
- **Neighborhood divergence**: O(R) where R = number of relations

## Migration Notes

Old saves (pre-tier) will automatically migrate:
- `tier` defaults to `1` (TIER_DOMAIN)
- `project` defaults to `"default"`
- `origin` defaults to `"kb"`
- `last_*_step` defaults to `-1000`

No manual intervention required!
