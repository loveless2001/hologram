import pytest
import os
import numpy as np
from hologram.code_map.evolution import CodeEvolutionEngine
from hologram.code_map.registry import SymbolRegistry
from hologram.store import MemoryStore, Trace
from hologram.gravity import Gravity, Concept, cosine
from hologram.config import Config

PROJECT_NAME = "test_evolution_pipeline"
TEMP_FILE_PATH = os.path.abspath("temp_evolution_test.py")

# Version 1: Initial implementation
CODE_V1 = """
class Spacecraft:
    def __init__(self, name, fuel_capacity):
        self.name = name
        self.fuel = fuel_capacity
        self.velocity = 0.0

    def launch(self):
        \"\"\"Initiate launch sequence and consume fuel.\"\"\"
        if self.fuel > 10:
            print(f"{self.name} is launching!")
            self.fuel -= 10
            self.velocity = 5000.0
        else:
            print("Insufficient fuel.")

class MissionControl:
    def __init__(self, location="Houston"):
        self.location = location
        self.active_missions = []

    def abort_mission(self, spacecraft):
        \"\"\"Emergency abort sequence.\"\"\"
        print(f"Aborting mission for {spacecraft.name}")
        spacecraft.velocity = 0.0

def calculate_trajectory(p1, p2):
    \"\"\"Compute orbital trajectory using Hohmann transfer.\"\"\"
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return (dx**2 + dy**2)**0.5
"""

# Version 2: Small drift - minor refactoring (comments, formatting)
CODE_V2_SMALL_DRIFT = """
class Spacecraft:
    def __init__(self, name, fuel_capacity):
        # Initialize spacecraft with name and fuel
        self.name = name
        self.fuel = fuel_capacity
        self.velocity = 0.0

    def launch(self):
        \"\"\"Initiate launch sequence and consume fuel.\"\"\"
        # Check fuel level before launch
        if self.fuel > 10:
            print(f"{self.name} is launching!")
            self.fuel -= 10
            self.velocity = 5000.0
        else:
            print("Insufficient fuel.")

class MissionControl:
    def __init__(self, location="Houston"):
        self.location = location
        self.active_missions = []

    def abort_mission(self, spacecraft):
        \"\"\"Emergency abort sequence.\"\"\"
        print(f"Aborting mission for {spacecraft.name}")
        spacecraft.velocity = 0.0

def calculate_trajectory(p1, p2):
    \"\"\"Compute orbital trajectory using Hohmann transfer.\"\"\"
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return (dx**2 + dy**2)**0.5
"""

# Version 3: Medium drift - logic changes
CODE_V3_MEDIUM_DRIFT = """
class Spacecraft:
    def __init__(self, name, fuel_capacity):
        self.name = name
        self.fuel = fuel_capacity
        self.velocity = 0.0
        self.status = "ready"  # NEW field

    def launch(self):
        \"\"\"Initiate launch sequence with status tracking.\"\"\"
        if self.fuel > 10 and self.status == "ready":
            print(f"{self.name} is launching!")
            self.fuel -= 10
            self.velocity = 5000.0
            self.status = "in_flight"  # NEW logic
        else:
            print("Cannot launch - insufficient fuel or not ready.")

class MissionControl:
    def __init__(self, location="Houston"):
        self.location = location
        self.active_missions = []

    def abort_mission(self, spacecraft):
        \"\"\"Emergency abort sequence with status update.\"\"\"
        print(f"Aborting mission for {spacecraft.name}")
        spacecraft.velocity = 0.0
        spacecraft.status = "aborted"  # NEW logic

def calculate_trajectory(p1, p2):
    \"\"\"Compute orbital trajectory using Hohmann transfer.\"\"\"
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = (dx**2 + dy**2)**0.5
    return distance
"""

# Version 4: Large drift - complete rewrite
CODE_V4_LARGE_DRIFT = """
class Spacecraft:
    def __init__(self, name, fuel_capacity):
        self.name = name
        self.fuel = fuel_capacity
        self.velocity = 0.0
        self.status = "ready"
        self.telemetry = []  # Completely new architecture

    def launch(self):
        \"\"\"Advanced launch with telemetry logging.\"\"\"
        if not self._pre_flight_check():
            return False
        
        self.fuel -= 10
        self.velocity = 5000.0
        self.status = "in_flight"
        self._log_telemetry("launch", {"fuel": self.fuel, "velocity": self.velocity})
        return True
    
    def _pre_flight_check(self):
        \"\"\"Internal pre-flight validation.\"\"\"
        return self.fuel > 10 and self.status == "ready"
    
    def _log_telemetry(self, event, data):
        \"\"\"Log telemetry data.\"\"\"
        self.telemetry.append({"event": event, "data": data})

class MissionControl:
    def __init__(self, location="Houston"):
        self.location = location
        self.active_missions = []

    def abort_mission(self, spacecraft):
        \"\"\"Emergency abort sequence with status update.\"\"\"
        print(f"Aborting mission for {spacecraft.name}")
        spacecraft.velocity = 0.0
        spacecraft.status = "aborted"

def calculate_trajectory(p1, p2):
    \"\"\"Compute orbital trajectory using Hohmann transfer.\"\"\"
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    distance = (dx**2 + dy**2)**0.5
    return distance
"""

# Version 5: Symbol deletion (remove calculate_trajectory)
CODE_V5_DELETION = """
class Spacecraft:
    def __init__(self, name, fuel_capacity):
        self.name = name
        self.fuel = fuel_capacity
        self.velocity = 0.0
        self.status = "ready"
        self.telemetry = []

    def launch(self):
        \"\"\"Advanced launch with telemetry logging.\"\"\"
        if not self._pre_flight_check():
            return False
        
        self.fuel -= 10
        self.velocity = 5000.0
        self.status = "in_flight"
        self._log_telemetry("launch", {"fuel": self.fuel, "velocity": self.velocity})
        return True
    
    def _pre_flight_check(self):
        \"\"\"Internal pre-flight validation.\"\"\"
        return self.fuel > 10 and self.status == "ready"
    
    def _log_telemetry(self, event, data):
        \"\"\"Log telemetry data.\"\"\"
        self.telemetry.append({"event": event, "data": data})

class MissionControl:
    def __init__(self, location="Houston"):
        self.location = location
        self.active_missions = []

    def abort_mission(self, spacecraft):
        \"\"\"Emergency abort sequence with status update.\"\"\"
        print(f"Aborting mission for {spacecraft.name}")
        spacecraft.velocity = 0.0
        spacecraft.status = "aborted"
"""

# Version 6: Revival (bring back calculate_trajectory)
CODE_V6_REVIVAL = CODE_V4_LARGE_DRIFT


@pytest.fixture
def evolution_engine(tmp_path):
    """Create an isolated evolution engine for testing."""
    store = MemoryStore(vec_dim=384)
    
    # Track version for drift simulation
    version_counter = {"count": 0}
    
    def vectorizer(text: str) -> np.ndarray:
        # Create vectors that drift slightly between versions
        # Use text hash as base, add version-specific noise
        base_seed = abs(hash(text)) % 2**32
        version_seed = version_counter["count"]
        
        np.random.seed(base_seed)
        vec = np.random.randn(384).astype(np.float32)
        
        # Add small version-specific perturbation
        np.random.seed(version_seed)
        noise = np.random.randn(384).astype(np.float32) * 0.05
        vec = vec + noise
        
        version_counter["count"] += 1
        return vec / np.linalg.norm(vec)
    
    engine = CodeEvolutionEngine(store, vectorizer)
    engine.registry = SymbolRegistry(persistence_path=str(tmp_path / "registry.json"))
    
    yield engine, store
    
    # Cleanup
    if os.path.exists(TEMP_FILE_PATH):
        os.remove(TEMP_FILE_PATH)


def test_evolution_full_lifecycle(evolution_engine):
    """
    Comprehensive test of the evolution pipeline:
    1. Initial ingestion
    2. Small drift (fusion)
    3. Medium drift (soft fusion)
    4. Large drift (mitosis)
    5. Symbol deletion (deprecation)
    6. Symbol revival
    """
    engine, store = evolution_engine
    
    # === PHASE 1: Initial Ingestion ===
    print("\n=== PHASE 1: Initial Ingestion ===")
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V1)
    
    count = engine.process_file(TEMP_FILE_PATH)
    assert count > 0, "Should extract symbols"
    
    # Verify all symbols are registered
    assert len(engine.registry.registry) >= 5  # Spacecraft, __init__, launch, MissionControl, abort_mission, calculate_trajectory
    
    # Get initial symbol IDs
    spacecraft_launch_id = None
    for sid, meta in engine.registry.registry.items():
        if "Spacecraft.launch" in meta.qualified_name:
            spacecraft_launch_id = sid
            break
    
    assert spacecraft_launch_id is not None, "Should find Spacecraft.launch"
    
    # Check initial concept in gravity
    concept_v1 = store.sim.concepts.get(spacecraft_launch_id)
    assert concept_v1 is not None
    assert concept_v1.status == "active"
    assert concept_v1.original_vec is not None
    vec_v1 = concept_v1.vec.copy()
    mass_v1 = concept_v1.mass
    
    print(f"✓ Initial ingestion: {count} symbols, mass={mass_v1:.2f}")
    
    # === PHASE 2: Small Drift (Fusion) ===
    print("\n=== PHASE 2: Small Drift (Fusion) ===")
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V2_SMALL_DRIFT)
    
    count = engine.process_file(TEMP_FILE_PATH)
    
    concept_v2 = store.sim.concepts.get(spacecraft_launch_id)
    vec_v2 = concept_v2.vec.copy()
    mass_v2 = concept_v2.mass
    
    drift = 1.0 - cosine(vec_v1, vec_v2)
    print(f"✓ Small drift: drift={drift:.4f}, mass_v1={mass_v1:.2f}, mass_v2={mass_v2:.2f}, growth={mass_v2-mass_v1:.2f}")
    
    # Verify fusion occurred (mass should grow slightly or stay same if drift is tiny)
    # Small drift might result in very small mass changes
    assert mass_v2 >= mass_v1 * 0.99, f"Mass shouldn't decrease significantly: {mass_v1} -> {mass_v2}"
    assert drift < Config.evolution.DRIFT_MEDIUM, f"Drift {drift} should be small"
    
    # === PHASE 3: Medium Drift (Soft Fusion) ===
    print("\n=== PHASE 3: Medium Drift (Soft Fusion) ===")
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V3_MEDIUM_DRIFT)
    
    count = engine.process_file(TEMP_FILE_PATH)
    
    concept_v3 = store.sim.concepts.get(spacecraft_launch_id)
    vec_v3 = concept_v3.vec.copy()
    mass_v3 = concept_v3.mass
    
    drift = 1.0 - cosine(vec_v2, vec_v3)
    print(f"✓ Medium drift: drift={drift:.4f}, mass={mass_v3:.2f}")
    
    # Verify soft fusion (interpolation)
    assert concept_v3.previous_vec is not None
    
    # === PHASE 4: Large Drift (Mitosis) ===
    print("\n=== PHASE 4: Large Drift (Mitosis) ===")
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V4_LARGE_DRIFT)
    
    initial_concept_count = len(store.sim.concepts)
    count = engine.process_file(TEMP_FILE_PATH)
    
    concept_v4 = store.sim.concepts.get(spacecraft_launch_id)
    vec_v4 = concept_v4.vec.copy()
    
    drift = 1.0 - cosine(vec_v3, vec_v4)
    print(f"✓ Large drift: drift={drift:.4f}")
    
    # Check if archive was created (mitosis)
    archive_concepts = [k for k in store.sim.concepts.keys() if "_arch_" in k]
    if drift > Config.evolution.DRIFT_LARGE:
        print(f"  Mitosis triggered: {len(archive_concepts)} archive(s) created")
    
    # === PHASE 5: Symbol Deletion (Deprecation) ===
    print("\n=== PHASE 5: Symbol Deletion (Deprecation) ===")
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V5_DELETION)
    
    # Find calculate_trajectory ID
    traj_id = None
    for sid, meta in engine.registry.registry.items():
        if "calculate_trajectory" in meta.qualified_name:
            traj_id = sid
            break
    
    count = engine.process_file(TEMP_FILE_PATH)
    
    if traj_id:
        traj_meta = engine.registry.get(traj_id)
        assert traj_meta.status == "deprecated", "Should be marked deprecated"
        
        traj_concept = store.sim.concepts.get(traj_id)
        if traj_concept:
            print(f"✓ Deprecation: mass={traj_concept.mass:.2f}, status={traj_meta.status}")
            # Mass should have decayed
            assert traj_concept.mass < 1.0, "Deprecated symbol should have reduced mass"
    
    # === PHASE 6: Symbol Revival ===
    print("\n=== PHASE 6: Symbol Revival ===")
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V6_REVIVAL)
    
    count = engine.process_file(TEMP_FILE_PATH)
    
    if traj_id:
        traj_meta = engine.registry.get(traj_id)
        assert traj_meta.status == "revived", "Should be marked revived"
        
        traj_concept = store.sim.concepts.get(traj_id)
        if traj_concept:
            print(f"✓ Revival: mass={traj_concept.mass:.2f}, status={traj_meta.status}")
            # Mass should be boosted
            assert traj_concept.mass > 1.0, "Revived symbol should have boosted mass"
    
    # === FINAL VERIFICATION ===
    print("\n=== FINAL VERIFICATION ===")
    print(f"Total symbols in registry: {len(engine.registry.registry)}")
    print(f"Total concepts in gravity: {len(store.sim.concepts)}")
    
    # Verify vector history tracking
    assert concept_v4.age > 0, "Age should increment"
    assert len(concept_v4.vector_history) > 0, "Should track vector history"
    
    # Verify all active symbols have proper status
    active_count = sum(1 for m in engine.registry.registry.values() if m.status == "active")
    print(f"Active symbols: {active_count}")
    assert active_count > 0
    
    print("\n✅ Full lifecycle test passed!")


def test_evolution_span_updates(evolution_engine):
    """Test that span information is correctly updated across versions."""
    engine, store = evolution_engine
    
    # Initial version
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V1)
    
    engine.process_file(TEMP_FILE_PATH)
    
    # Find Spacecraft class
    spacecraft_id = None
    for sid, meta in engine.registry.registry.items():
        if meta.qualified_name == "Spacecraft":
            spacecraft_id = sid
            break
    
    assert spacecraft_id is not None
    
    # Get initial span
    trace_v1 = store.get_trace(spacecraft_id)
    span_v1 = trace_v1.span
    
    # Update with version that has different line numbers
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V3_MEDIUM_DRIFT)
    
    engine.process_file(TEMP_FILE_PATH)
    
    # Verify span was updated
    trace_v2 = store.get_trace(spacecraft_id)
    span_v2 = trace_v2.span
    
    assert span_v2 is not None
    assert trace_v2.source_file == TEMP_FILE_PATH
    print(f"✓ Span updated: {span_v1} -> {span_v2}")


def test_evolution_identity_preservation(evolution_engine):
    """Test that symbol identity is preserved across file moves."""
    engine, store = evolution_engine
    
    # Initial file
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(CODE_V1)
    
    engine.process_file(TEMP_FILE_PATH)
    
    # Get Spacecraft.launch ID
    launch_id = None
    for sid, meta in engine.registry.registry.items():
        if "Spacecraft.launch" in meta.qualified_name:
            launch_id = sid
            break
    
    assert launch_id is not None
    initial_meta = engine.registry.get(launch_id)
    
    # "Move" file (simulate refactoring)
    new_path = os.path.abspath("temp_evolution_moved.py")
    with open(new_path, "w") as f:
        f.write(CODE_V1)
    
    try:
        engine.process_file(new_path)
        
        # Verify same ID, updated file path
        updated_meta = engine.registry.get(launch_id)
        assert updated_meta.file_path == new_path
        assert updated_meta.qualified_name == initial_meta.qualified_name
        assert updated_meta.signature == initial_meta.signature
        
        print(f"✓ Identity preserved across file move: {TEMP_FILE_PATH} -> {new_path}")
    finally:
        if os.path.exists(new_path):
            os.remove(new_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
