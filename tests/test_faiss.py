import numpy as np
import sys
import os

# Add the current directory to sys.path so we can import hologram
sys.path.append(os.getcwd())

try:
    import faiss
    print(f"FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    try:
        print(f"FAISS path: {faiss.__file__}")
    except:
        pass
except ImportError:
    print("Failed to import faiss directly")

try:
    from hologram.store import VectorIndex
    print("Successfully imported VectorIndex from hologram.store")
except ImportError as e:
    print(f"Failed to import VectorIndex: {e}")
    sys.exit(1)

def test_faiss_integration():
    dim = 128
    print(f"Initializing VectorIndex with dim={dim}...")
    try:
        index = VectorIndex(dim=dim, use_gpu=True)
        print("VectorIndex initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize VectorIndex: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create dummy data
    key = "test_item"
    vec = np.random.rand(dim).astype('float32')
    
    print("Upserting vector...")
    try:
        index.upsert(key, vec)
        print("Upsert successful.")
    except Exception as e:
        print(f"Upsert failed: {e}")
        return

    print("Searching vector...")
    try:
        results = index.search(vec, top_k=1)
        print(f"Search results: {results}")
        if results and results[0][0] == key:
            print("Test PASSED: Retrieved correct key.")
        else:
            print("Test FAILED: Did not retrieve correct key.")
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    test_faiss_integration()
