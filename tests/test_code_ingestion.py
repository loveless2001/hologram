import pytest
from fastapi.testclient import TestClient
import os
import shutil
from hologram.server import app

client = TestClient(app)
PROJECT_NAME = "test_code_mapping_layer"

# A richer code snippet with classes, docstrings, and distinct concepts
RICH_CODE_CONTENT = """
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

TEMP_FILE_PATH = os.path.abspath("temp_space_mission.py")

@pytest.fixture
def setup_teardown():
    # Setup
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(RICH_CODE_CONTENT)
    
    # Clean memory for project
    client.post(f"/reset/{PROJECT_NAME}?confirm=true")
    
    yield
    
    # Teardown
    if os.path.exists(TEMP_FILE_PATH):
        os.remove(TEMP_FILE_PATH)
    # Optional: Reset again to clean up
    client.post(f"/reset/{PROJECT_NAME}?confirm=true")


def test_code_ingestion_and_query(setup_teardown):
    # 1. Ingest
    payload = {
        "project": PROJECT_NAME,
        "path": TEMP_FILE_PATH,
        "tier": 1
    }
    
    res = client.post("/ingest/code", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "concepts_extracted" in data
    assert data["concepts_extracted"] > 0
    
    # 2. Query - Class Name
    q_payload = {
        "project": PROJECT_NAME,
        "text": "Spacecraft",
        "top_k": 3
    }
    res = client.post("/query/code", json=q_payload)
    assert res.status_code == 200
    results = res.json().get("results", [])
    assert len(results) > 0
    # Check if file/span is present (Evolution Engine updates should ensure this)
    assert results[0]["file"] == TEMP_FILE_PATH
    assert results[0]["span"] is not None
    
    # 3. Query - Semantic (Docstring)
    q_payload = {
        "project": PROJECT_NAME,
        "text": "launch sequence",
        "top_k": 3
    }
    res = client.post("/query/code", json=q_payload)
    assert res.status_code == 200
    results = res.json().get("results", [])
    assert len(results) > 0
    # Should match 'launch' function
    assert "launch" in results[0]["snippet"] or "launch" in results[0]["concept"]
