"""
MG Scorer Chaos Tests
=====================

Stress testing and edge case verification for the Memory Gravity Scorer.
"""

import pytest
import numpy as np
import threading
import time
from hologram.mg_scorer import mg_score, coherence, curvature, semantic_entropy, collapse_risk

class TestMGStability:
    """Test numerical stability and edge cases."""

    def test_nan_vectors(self):
        """CHAOS: Vectors containing NaNs should not crash (or handle gracefully)."""
        # Current implementation might propagate NaNs. Let's see.
        vecs = [np.array([1.0, np.nan, 0.0]), np.array([0.0, 1.0, 0.0])]
        # Should probably raise ValueError or return NaN score, but not crash process
        try:
            score = mg_score(vecs)
            # If it returns, check if values are NaN
            assert np.isnan(score.coherence) or score.coherence >= 0
        except ValueError:
            pass  # Acceptable to reject NaNs

    def test_inf_vectors(self):
        """CHAOS: Vectors containing Infinity."""
        vecs = [np.array([1.0, np.inf, 0.0]), np.array([0.0, 1.0, 0.0])]
        try:
            score = mg_score(vecs)
        except (ValueError, OverflowError):
            pass

    def test_zero_vectors(self):
        """CHAOS: All-zero vectors (norm=0)."""
        # Coherence uses cosine similarity which divides by norm.
        # Implementation adds 1e-12 to norm, so it should be stable.
        vecs = [np.zeros(5), np.zeros(5)]
        score = mg_score(vecs)
        # Cosine of 0-vecs is undefined but handled.
        # If handled as 0 similarity -> distance 1 -> coherence 0.
        # Or handled as identical -> coherence 1.
        # Let's check behavior.
        assert 0.0 <= score.coherence <= 1.0

    def test_single_vector(self):
        """CHAOS: Single vector input."""
        vecs = [np.random.rand(5)]
        score = mg_score(vecs)
        assert score.coherence == 1.0  # Convention
        assert score.curvature == 1.0  # Convention
        assert score.entropy == 0.0    # No variation

    def test_empty_input(self):
        """CHAOS: Empty input list."""
        with pytest.raises(ValueError):
            mg_score([])

class TestMGConcurrency:
    """Test thread safety of scoring functions."""

    def test_concurrent_scoring(self):
        """CHAOS: Multiple threads scoring different datasets."""
        errors = []
        
        def worker():
            try:
                for _ in range(100):
                    vecs = np.random.rand(10, 128)
                    score = mg_score(vecs)
                    assert 0 <= score.coherence <= 1
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(errors) == 0, f"Concurrent scoring failed: {errors}"

class TestMGScale:
    """Test scaling behavior."""

    def test_large_batch(self):
        """CHAOS: Score a large batch (10k vectors)."""
        # 10k vectors of dim 64
        vecs = np.random.rand(10000, 64)
        start = time.time()
        score = mg_score(vecs)
        duration = time.time() - start
        
        print(f"Scored 10k vectors in {duration:.4f}s")
        assert duration < 30.0  # O(N^2) coherence is slow for 10k
        assert 0 <= score.coherence <= 1

    def test_high_dimension(self):
        """CHAOS: Score high-dimensional vectors (dim 4096)."""
        vecs = np.random.rand(100, 4096)
        score = mg_score(vecs)
        assert 0 <= score.coherence <= 1

if __name__ == "__main__":
    pytest.main([__file__])
