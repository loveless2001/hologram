# tests/test_manifold.py
import numpy as np
from hologram.manifold import LatentManifold

class MockEncoder:
    def encode(self, text):
        # Return a non-normalized vector
        return np.array([1.0, 2.0, 3.0])
        
    def encode_path(self, path):
        return np.array([4.0, 5.0, 6.0])

def test_projection():
    manifold = LatentManifold(dim=3)
    vec = np.array([1.0, 1.0, 1.0])
    proj = manifold.project(vec)
    
    # Check normalization
    assert np.isclose(np.linalg.norm(proj), 1.0)
    
def test_align_text():
    manifold = LatentManifold(dim=3)
    encoder = MockEncoder()
    
    vec = manifold.align_text("test", encoder)
    assert np.isclose(np.linalg.norm(vec), 1.0)
    
def test_align_image():
    manifold = LatentManifold(dim=3)
    encoder = MockEncoder()
    
    vec = manifold.align_image("path/to/img", encoder)
    assert np.isclose(np.linalg.norm(vec), 1.0)

if __name__ == "__main__":
    test_projection()
    test_align_text()
    test_align_image()
    print("LatentManifold tests passed!")
