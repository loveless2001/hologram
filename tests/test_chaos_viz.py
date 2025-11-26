import pytest
import threading
import time
import numpy as np
from fastapi.testclient import TestClient
from api_server.main import app, load_kb, memory

client = TestClient(app)

class TestChaosViz:
    def setup_method(self):
        # Ensure clean state
        load_kb(None)
        # Initialize with gravity
        load_kb("chaos_test_kb")
        
    def test_concurrent_viz_and_updates(self):
        """Test /viz-data endpoint while concurrently adding concepts."""
        
        errors = []
        stop_event = threading.Event()
        
        def read_viz():
            while not stop_event.is_set():
                try:
                    resp = client.get("/viz-data")
                    assert resp.status_code == 200
                    data = resp.json()
                    assert "points" in data
                    assert "labels" in data
                    assert len(data["points"]) == len(data["labels"])
                except Exception as e:
                    errors.append(e)
                    break
        
        def write_concepts():
            for i in range(50):
                try:
                    client.post("/chat", json={"message": f"concept_{i}", "kb_name": "chaos_test_kb"})
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(e)
                    break
                    
        reader = threading.Thread(target=read_viz)
        writer = threading.Thread(target=write_concepts)
        
        reader.start()
        writer.start()
        
        writer.join()
        stop_event.set()
        reader.join()
        
        assert not errors, f"Errors occurred: {errors}"
        
    def test_viz_empty_state(self):
        """Test /viz-data when no KB is loaded."""
        load_kb(None)
        resp = client.get("/viz-data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["points"] == []
        assert data["labels"] == []
