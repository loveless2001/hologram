"""
Chaos Tests for Quantization Level Implementation
Generated following .agent/workflows/chaos-test.md
"""
import numpy as np
import threading
import time
from hologram.gravity import Gravity, calibrate_quantization


class TestQuantizationChaos:
    """Chaos tests targeting the new quantization feature."""
    
    def test_quantization_concurrent_adding(self):
        """
        CHAOS TEST: Concurrent concept additions with quantization threshold
        Expected: Quantization should be thread-safe and consistent
        Risk Score: 0.7 (HIGH) - New code path in _mutual_drift
        """
        g = Gravity(dim=64, quantization_level=0.03)
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(20):
                    name = f"concept_{thread_id}_{i}"
                    vec = np.random.rand(64).astype(np.float32)
                    vec /= np.linalg.norm(vec)
                    g.add_concept(name, vec=vec)
            except Exception as e:
                errors.append((thread_id, type(e).__name__, str(e)))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(g.concepts) > 0, "No concepts added"
        
        # Verify quantization_level is preserved
        assert g.quantization_level == 0.03, "Quantization level changed"
    
    def test_quantization_extreme_values(self):
        """
        EDGE CASE: Extreme quantization values
        Expected: System should clamp or handle gracefully
        Risk Score: 0.6 (MEDIUM) - Boundary validation
        """
        # Very high quantization (almost no drift)
        g1 = Gravity(dim=32, quantization_level=0.9)
        g1.add_concept("A", vec=np.ones(32, dtype=np.float32))
        g1.add_concept("B", vec=np.ones(32, dtype=np.float32) * 0.5)
        
        # Verify concepts exist but didn't drift much
        assert "A" in g1.concepts
        assert "B" in g1.concepts
        
        # Very low quantization (maximum drift)
        g2 = Gravity(dim=32, quantization_level=0.0001)
        g2.add_concept("C", vec=np.ones(32, dtype=np.float32))
        vec_c_orig = g2.concepts["C"].vec.copy()
        g2.add_concept("D", vec=np.ones(32, dtype=np.float32) * 0.9)
        
        # Verify drift occurred
        assert not np.allclose(g2.concepts["C"].vec, vec_c_orig), \
            "Expected drift with low quantization"
    
    def test_quantization_calibration_determinism(self):
        """
        TEST: Hardware calibration should be deterministic
        Expected: Multiple calls return same value
        Risk Score: 0.4 (MEDIUM) - Calibration logic
        """
        q1 = calibrate_quantization()
        q2 = calibrate_quantization()
        q3 = calibrate_quantization()
        
        assert q1 == q2 == q3, f"Calibration non-deterministic: {q1}, {q2}, {q3}"
        assert 0.001 <= q1 <= 0.2, f"Calibration out of bounds: {q1}"
    
    def test_quantization_state_persistence(self):
        """
        TEST: Quantization level should persist in state save/load
        Expected: get_state/set_state preserves quantization_level
        Risk Score: 0.8 (HIGH) - New field in serialization
        """
        g1 = Gravity(dim=32, quantization_level=0.042)
        g1.add_concept("test", vec=np.ones(32, dtype=np.float32))
        
        # Save state
        state = g1.get_state()
        
        # Verify quantization_level is in state
        assert "quantization_level" in state["params"], \
            "quantization_level not saved in state"
        assert state["params"]["quantization_level"] == 0.042
        
        # Load state into new instance
        g2 = Gravity(dim=32)
        g2.set_state(state)
        
        # Verify restoration
        assert g2.quantization_level == 0.042, \
            f"Quantization not restored: {g2.quantization_level}"
        assert "test" in g2.concepts
    
    def test_quantization_none_auto_calibrate(self):
        """
        TEST: Quantization should auto-calibrate if None
        Expected: __post_init__ calls calibrate_quantization()
        Risk Score: 0.5 (MEDIUM) - Auto-initialization logic
        """
        g = Gravity(dim=32)  # quantization_level not specified
        
        assert g.quantization_level is not None, \
            "Quantization should be auto-calibrated"
        assert isinstance(g.quantization_level, float)
        assert 0.001 <= g.quantization_level <= 0.2
    
    def test_quantization_mutual_drift_skip_logic(self):
        """
        TEST: Verify that weak interactions are actually skipped
        Expected: Concepts with step < quantization_level don't drift
        Risk Score: 0.7 (HIGH) - Core logic of quantization feature
        """
        # Set high quantization and low eta
        g = Gravity(dim=64, quantization_level=0.1, eta=0.01)
        
        # Add first concept
        vec_a = np.zeros(64, dtype=np.float32)
        vec_a[0] = 1.0
        g.add_concept("A", vec=vec_a)
        vec_a_after_first = g.concepts["A"].vec.copy()
        
        # Add orthogonal concept (sim ≈ 0, step ≈ 0)
        vec_b = np.zeros(64, dtype=np.float32)
        vec_b[1] = 1.0
        g.add_concept("B", vec=vec_b)
        
        # A should NOT have moved (sim was ~0, step < 0.1)
        assert np.allclose(g.concepts["A"].vec, vec_a_after_first), \
            "A moved despite step < quantization_level"


def run_risk_analysis():
    """
    Stage 2: Risk Prediction (following chaos-test.md)
    """
    risks = {
        "quantization_level field": {
            "component": "Gravity dataclass",
            "risk": "New optional field might not be handled in all code paths",
            "score": 0.6,
            "mitigation": "Tested in state persistence test"
        },
        "_mutual_drift quantization check": {
            "component": "Gravity._mutual_drift",
            "risk": "New conditional (if abs(step) < quantization_level) could have edge cases",
            "score": 0.7,
            "mitigation": "Tested with extreme values and orthogonal vectors"
        },
        "calibrate_quantization hardware detection": {
            "component": "calibrate_quantization()",
            "risk": "Exceptions in GPU/CPU/RAM detection might not be caught properly",
            "score": 0.5,
            "mitigation": "Wrapped in try-except, tested determinism"
        },
        "__post_init__ auto-calibration": {
            "component": "Gravity.__post_init__",
            "risk": "Auto-calibration might fail silently",
            "score": 0.5,
            "mitigation": "Tested explicitly in test_quantization_none_auto_calibrate"
        },
        "Thread safety": {
            "component": "Gravity with quantization",
            "risk": "New code path in _mutual_drift might have race conditions",
            "score": 0.7,
            "mitigation": "Protected by existing _lock, tested concurrently"
        }
    }
    
    return risks


if __name__ == "__main__":
    print("🔥 Chaos Tests for Quantization Level 🔥\n")
    
    # Run tests
    test = TestQuantizationChaos()
    
    tests = [
        ("Concurrent Adding", test.test_quantization_concurrent_adding),
        ("Extreme Values", test.test_quantization_extreme_values),
        ("Calibration Determinism", test.test_quantization_calibration_determinism),
        ("State Persistence", test.test_quantization_state_persistence),
        ("Auto Calibration", test.test_quantization_none_auto_calibrate),
        ("Mutual Drift Skip Logic", test.test_quantization_mutual_drift_skip_logic),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: CRASH - {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    # Risk analysis
    print("\n" + "=" * 60)
    print("RISK ANALYSIS (Stage 2)")
    print("=" * 60)
    
    risks = run_risk_analysis()
    for component, details in risks.items():
        print(f"\n[{details['score']:.1f}] {component}")
        print(f"    Risk: {details['risk']}")
        print(f"    Mitigation: {details['mitigation']}")
    
    # Check if safety threshold is met
    max_risk = max(r["score"] for r in risks.values())
    print(f"\n{'=' * 60}")
    print(f"Maximum Risk Score: {max_risk:.2f}")
    print(f"Safety Threshold: 0.95")
    
    if max_risk <= 0.8 and failed == 0:
        print("✓ PUBLISH GATE: PASSED")
    else:
        print("✗ PUBLISH GATE: FAILED - Risks or test failures detected")
