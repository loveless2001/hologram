# tests/test_mitosis_geometry.py
import numpy as np
from hologram.gravity import Gravity

def test_geometry_based_mitosis():
    print("--- Testing Geometry-Based Mitosis ---")
    g = Gravity(dim=64, seed=42)
    
    # 1. Create a "polysemous" concept: "bank"
    # It will be connected to two distinct clusters: "finance" and "river"
    
    # Cluster 1: Finance (random vectors around a centroid)
    finance_centroid = np.random.rand(64).astype('float32')
    finance_centroid /= np.linalg.norm(finance_centroid)
    
    finance_words = ["money", "loan", "deposit", "interest"]
    for w in finance_words:
        vec = finance_centroid + np.random.normal(0, 0.1, 64).astype('float32')
        vec /= np.linalg.norm(vec)
        g.add_concept(w, vec=vec)
        
    # Cluster 2: River (random vectors around a DIFFERENT centroid)
    # Make sure it's far away (orthogonal-ish)
    river_centroid = np.random.rand(64).astype('float32')
    # Flip signs to ensure distance
    river_centroid = -river_centroid 
    river_centroid /= np.linalg.norm(river_centroid)
    
    river_words = ["water", "stream", "fish", "flow"]
    for w in river_words:
        vec = river_centroid + np.random.normal(0, 0.1, 64).astype('float32')
        vec /= np.linalg.norm(vec)
        g.add_concept(w, vec=vec)
        
    # Add "bank" in the middle (average of both)
    bank_vec = (finance_centroid + river_centroid) / 2.0
    bank_vec /= np.linalg.norm(bank_vec)
    g.add_concept("bank", vec=bank_vec)
    
    # Create strong relations to both clusters to simulate tension
    for w in finance_words + river_words:
        key = (min("bank", w), max("bank", w))
        g.relations[key] = 0.8  # Strong connection
        
    # 2. Trigger Mitosis Check
    # Threshold 0.4 should be exceeded if clusters are distinct
    print("Checking mitosis for 'bank'...")
    occurred = g.check_mitosis("bank", threshold=0.4)
    
    assert occurred, "Mitosis should have occurred for bimodal distribution"
    assert "bank" not in g.concepts, "Original 'bank' should be removed"
    assert "bank_1" in g.concepts, "'bank_1' should exist"
    assert "bank_2" in g.concepts, "'bank_2' should exist"
    
    print("✓ Mitosis occurred successfully")
    
    # 3. Verify Separation
    # One sibling should be closer to finance, one to river
    vec1 = g.concepts["bank_1"].vec
    vec2 = g.concepts["bank_2"].vec
    
    dist1_fin = np.dot(vec1, finance_centroid)
    dist1_riv = np.dot(vec1, river_centroid)
    
    dist2_fin = np.dot(vec2, finance_centroid)
    dist2_riv = np.dot(vec2, river_centroid)
    
    print(f"Bank_1 -> Finance: {dist1_fin:.3f}, River: {dist1_riv:.3f}")
    print(f"Bank_2 -> Finance: {dist2_fin:.3f}, River: {dist2_riv:.3f}")
    
    # One should be high/low, the other low/high
    assert (dist1_fin > dist1_riv and dist2_riv > dist2_fin) or \
           (dist1_riv > dist1_fin and dist2_fin > dist2_riv), \
           "Siblings did not separate into distinct clusters"
           
    print("✓ Siblings separated correctly")

def test_no_mitosis_unimodal():
    print("\n--- Testing Unimodal Case (No Mitosis) ---")
    g = Gravity(dim=64, seed=42)
    
    # Create a single tight cluster
    centroid = np.random.rand(64).astype('float32')
    centroid /= np.linalg.norm(centroid)
    
    words = ["apple", "banana", "fruit", "pear", "orange"]
    for w in words:
        vec = centroid + np.random.normal(0, 0.05, 64).astype('float32')
        vec /= np.linalg.norm(vec)
        g.add_concept(w, vec=vec)
        
    # Add "food" connected to all
    g.add_concept("food", vec=centroid)
    for w in words:
        key = (min("food", w), max("food", w))
        g.relations[key] = 0.8
        
    occurred = g.check_mitosis("food", threshold=0.4)
    assert not occurred, "Mitosis should NOT occur for unimodal distribution"
    print("✓ Mitosis correctly avoided for unimodal cluster")

if __name__ == "__main__":
    test_geometry_based_mitosis()
    test_no_mitosis_unimodal()
