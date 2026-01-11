# tests/test_search_relations.py
import pytest
import os
from fastapi.testclient import TestClient
from hologram.server import app, hologram_instances

# Use absolute path for test data or mock it
KB_PATH = os.path.abspath("tests/data/relativity.txt")

@pytest.fixture
def client():
    # Setup
    hologram_instances.clear()
    with TestClient(app) as c:
        yield c
    # Teardown
    hologram_instances.clear()

def test_search_with_relations(client):
    print("=" * 60)
    print("Testing Enhanced /search Endpoint with Relations")
    print("=" * 60)
    
    # 0. Create dummy KB file if not exists
    if not os.path.exists(KB_PATH):
        os.makedirs(os.path.dirname(KB_PATH), exist_ok=True)
        with open(KB_PATH, "w") as f:
            f.write("The speed of light is constant. Time dilation occurs near light speed.")
    
    # 1. Load KB
    print("\n1. Loading KB...")
    # Use /ingest to load content (since /chat load is deprecated/missing)
    with open(KB_PATH, "r") as f:
        kb_text = f.read()
        
    response = client.post("/ingest", json={
        "project": "default",
        "text": kb_text,
        "origin": "kb",
        "metadata": {"source": "relativity.txt"}
    })
    print(f"   Status: {response.status_code}")
    # We assert success here to catch errors early
    assert response.status_code == 200
    
    # 2. Search with relations
    print(f"\n2. Searching for 'speed of light' with relations...")
    response = client.post("/query", json={
        "project": "default",
        "text": "speed of light",
        "top_k": 3
    })
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        print(f"Body: {response.text}")
        
    assert response.status_code == 200
    data = response.json()
    print(f"\n   Query: '{data['query']}'")
    # Response structure: nodes, edges, glyphs, trajectory_steps
    # Map to test expectation (it used to expect "results")
    if "nodes" in data:
         results = data["nodes"]
    else:
         results = data.get("results", [])
         
    print(f"   Found {len(results)} results\n")
    
    assert len(results) > 0
    
    # 3. Search for a different keyword
    print("\n3. Searching for 'time dilation' with relations...")
    response = client.post("/query", json={
        "project": "default",
        "text": "time dilation",
        "top_k": 3
    })
    
    assert response.status_code == 200
    data = response.json()
    
    if "nodes" in data:
         results = data["nodes"]
    else:
         results = data.get("results", [])
         
    assert len(results) > 0

    print("\nTest Complete!")

