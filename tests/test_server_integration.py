import subprocess
import sys
import time
import requests
import os
import signal
import pytest
from pathlib import Path

# Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8001  # Use a different port to avoid conflicts
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
PROJECT_NAME = "test_integration_project"

@pytest.fixture(scope="module")
def server_process():
    """Start the Hologram server in a background process."""
    # Ensure we use the venv python
    python_executable = sys.executable
    
    # Start server
    cmd = [
        python_executable, "-m", "uvicorn",
        "hologram.server:app",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT)
    ]
    
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    
    # Wait for server to start
    max_retries = 20
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("Server started successfully!")
                break
        except requests.ConnectionError:
            time.sleep(0.5)
    else:
        # Server failed to start
        proc.kill()
        stdout, stderr = proc.communicate()
        print(f"Server stdout: {stdout.decode()}")
        print(f"Server stderr: {stderr.decode()}")
        raise RuntimeError("Server failed to start")
        
    yield proc
    
    # Cleanup
    print("Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

def test_health_check(server_process):
    """Verify server is running."""
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert "Hologram Server" in data["service"]

def test_ingest_domain_concept(server_process):
    """Test ingesting a Tier 1 domain concept."""
    payload = {
        "project": PROJECT_NAME,
        "text": "def calculate_fibonacci(n): return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
        "path": "/src/utils/math.py",
        "origin": "code",
        "tier": 1,
        "metadata": {"language": "python"}
    }
    
    resp = requests.post(f"{BASE_URL}/ingest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["project"] == PROJECT_NAME
    assert data["tier"] == 1
    assert "trace_id" in data

def test_auto_ingest_system_concepts(server_process):
    """Verify system concepts were auto-ingested (Tier 2)."""
    # Get memory summary
    resp = requests.get(f"{BASE_URL}/memory/{PROJECT_NAME}")
    assert resp.status_code == 200
    data = resp.json()
    
    # Check Tier 2 count (actual count is around 26-27)
    print(f"Tier 2 concepts found: {data['tier2_count']}")
    assert data["tier2_count"] >= 25

def test_query_memory(server_process):
    """Test querying the memory."""
    payload = {
        "project": PROJECT_NAME,
        "text": "fibonacci calculation",
        "top_k": 3
    }
    
    resp = requests.post(f"{BASE_URL}/query", json=payload)
    if resp.status_code != 200:
        print(f"Query failed: {resp.text}")
    assert resp.status_code == 200
    data = resp.json()
    
    assert len(data["nodes"]) > 0
    assert data["query"] == "fibonacci calculation"

def test_save_and_load(server_process):
    """Test saving and loading memory."""
    # Save
    resp = requests.post(f"{BASE_URL}/save/{PROJECT_NAME}")
    assert resp.status_code == 200
    save_path = resp.json()["path"]
    assert os.path.exists(save_path)
    
    # Load (into a new project name to verify)
    NEW_PROJECT = PROJECT_NAME + "_loaded"
    resp = requests.post(f"{BASE_URL}/load/{NEW_PROJECT}", params={"path": save_path})
    assert resp.status_code == 200
    
    # Verify loaded data
    resp = requests.get(f"{BASE_URL}/memory/{NEW_PROJECT}")
    assert resp.status_code == 200
    data = resp.json()
    
    # Should match original counts
    assert data["tier2_count"] >= 25
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
        # Try to remove parent dir if empty
        try:
            os.rmdir(os.path.dirname(save_path))
        except:
            pass

if __name__ == "__main__":
    # Manual run if executed directly
    # This part is just for quick debugging without pytest
    pass
