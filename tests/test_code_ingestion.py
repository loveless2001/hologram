import requests
import time
import json
import sys
import os

BASE_URL = "http://127.0.0.1:8000"
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

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def wait_for_server():
    print("⏳ Waiting for Hologram Server...")
    for _ in range(10):
        try:
            requests.get(BASE_URL)
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False

def run_test():
    if not wait_for_server():
        print("❌ Server is not running. Start it first!")
        return

    # Create temp file
    with open(TEMP_FILE_PATH, "w") as f:
        f.write(RICH_CODE_CONTENT)
    print(f"📄 Created temp file: {TEMP_FILE_PATH}")

    try:
        # 1. Reset Project (Clean Slate)
        print_header("1. CLEANSING MEMORY")
        requests.post(f"{BASE_URL}/reset/{PROJECT_NAME}?confirm=true")
        print(f"   Cleared project: {PROJECT_NAME}")

        # 2. Ingest the Code File using NEW Endpoint
        print_header("2. INGESTING CODE (Code Mapping Layer)")
        
        payload = {
            "project": PROJECT_NAME,
            "path": TEMP_FILE_PATH,
            "tier": 1
        }
        
        start = time.time()
        res = requests.post(f"{BASE_URL}/ingest/code", json=payload)
        duration = time.time() - start
        
        if res.status_code == 200:
            count = res.json().get("concepts_extracted")
            print(f"   ✅ Ingested successfully in {duration:.2f}s")
            print(f"   🧩 Concepts Extracted: {count}")
        else:
            print(f"   ❌ Ingestion Failed: {res.text}")
            return

        # 3. Test Retrievals using NEW Endpoint
        print_header("3. VALIDATING RETRIEVAL (/query/code)")
        
        queries = [
            ("Spacecraft Class", "Spacecraft"),
            ("Launch Function", "launch sequence"),  # Semantic search matches docstring
            ("Emergency Handling", "abort_mission"),
            ("Orbital Math", "trajectory calculation")
        ]

        for label, search_term in queries:
            print(f"\n🔍 Query: '{search_term}' ({label})")
            
            q_payload = {
                "project": PROJECT_NAME,
                "text": search_term,
                "top_k": 3
            }
            res = requests.post(f"{BASE_URL}/query/code", json=q_payload)
            
            if res.status_code == 200:
                results = res.json().get("results", [])
                
                if results:
                    best = results[0]
                    print(f"   ✅ MATCHED: {best['concept']}")
                    print(f"      Score: {best['score']:.3f}")
                    print(f"      File: {best['file']}")
                    print(f"      Span: {best['span']}")
                else:
                    print(f"   ❌ NO RESULTS FOUND")
            else:
                print(f"   ❌ Error: {res.status_code}")
                
    finally:
        # Cleanup
        if os.path.exists(TEMP_FILE_PATH):
            os.remove(TEMP_FILE_PATH)
            print(f"\n🗑️  Deleted temp file: {TEMP_FILE_PATH}")

if __name__ == "__main__":
    run_test()
