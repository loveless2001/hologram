from fastapi.testclient import TestClient
from api_server.main import app
import os

client = TestClient(app)

def test_kb_lifecycle():
    # 1. List KBs (should be empty initially or contain existing)
    response = client.get("/kbs")
    assert response.status_code == 200
    initial_kbs = response.json()["kbs"]
    
    # 2. Upload a test KB
    test_content = "Holographic memory is a distributed storage system."
    files = {"file": ("test_kb.txt", test_content, "text/plain")}
    response = client.post("/kbs/upload", files=files)
    assert response.status_code == 200
    assert response.json()["filename"] == "test_kb.txt"
    
    # 3. Verify it appears in list
    response = client.get("/kbs")
    assert "test_kb.txt" in response.json()["kbs"]
    
    # 4. Chat with it
    # First request switches to it
    payload = {"message": "what is holographic memory", "kb_name": "test_kb.txt"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "reply" in response.json()
    print(f"Reply: {response.json()['reply']}")
    
    # 5. Delete it
    response = client.delete("/kbs/test_kb.txt")
    assert response.status_code == 200
    
    # 6. Verify gone
    response = client.get("/kbs")
    assert "test_kb.txt" not in response.json()["kbs"]

if __name__ == "__main__":
    try:
        test_kb_lifecycle()
        print("✅ API Tests Passed!")
    except Exception as e:
        print(f"❌ API Tests Failed: {e}")
