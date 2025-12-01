
import pytest
import time
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.api import Hologram

@pytest.fixture
def small_store(tmp_path):
    db_path = tmp_path / "test_perf.db"
    holo = Hologram.init(encoder_mode="minilm", use_gravity=True)
    # Add some data
    for i in range(20):
        holo.add_text("g", f"This is trace {i} about gravity and physics.", trace_id=f"t{i}")
    holo.save(str(db_path))
    return str(db_path)

def test_search_latency(small_store):
    """Ensure search is under 10ms for small store (cached)."""
    holo = Hologram.load(small_store, encoder_mode="minilm", use_gravity=True)
    
    # Warmup
    holo.search_text("gravity")
    
    start = time.time()
    for _ in range(10):
        holo.search_text("gravity")
    duration = time.time() - start
    avg_ms = (duration / 10) * 1000
    
    print(f"Avg search latency: {avg_ms:.2f}ms")
    # Threshold: 50ms (generous, usually <1ms with FAISS)
    assert avg_ms < 50

def test_viz_latency(small_store):
    """Ensure viz projection is fast (cached)."""
    holo = Hologram.load(small_store, encoder_mode="minilm", use_gravity=True)
    
    # First call (compute)
    start = time.time()
    holo.field_state()
    first_ms = (time.time() - start) * 1000
    
    # Second call (cached)
    start = time.time()
    holo.field_state()
    second_ms = (time.time() - start) * 1000
    
    print(f"Viz latency: 1st={first_ms:.2f}ms, 2nd={second_ms:.2f}ms")
    
    # 2nd call should be very fast
    assert second_ms < 10
