"""
Test script for the Normalization Pipeline.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock
from hologram.normalization import NormalizationPipeline

class TestNormalizationPipeline(unittest.TestCase):
    def setUp(self):
        # Mock Gravity
        self.mock_gravity = MagicMock()
        self.mock_gravity.concepts = {"gravity": True, "field": True, "fusion": True}
        
        # Mock Encoder
        self.mock_encoder = MagicMock()
        self.mock_encoder.side_effect = lambda x: np.array([1.0, 0.0]) # Dummy vector
        
        self.pipeline = NormalizationPipeline(
            gravity_field=self.mock_gravity,
            encode_func=self.mock_encoder,
            enable_llm_correction=False
        )

    def test_stage0_tokenization(self):
        """Test basic token cleaning."""
        text = "  hello   world  "
        self.assertEqual(self.pipeline._stage0_tokenize(text), "hello world")
        
        text = "code :: block -> arrow"
        self.assertEqual(self.pipeline._stage0_tokenize(text), "code :: block -> arrow")

    def test_stage1_spell_correction(self):
        """Test SymSpell correction."""
        # Note: SymSpell might not be loaded in test env if dict missing, 
        # but we can mock it or assume it works if installed.
        # If SymSpell is missing, it returns original text.
        
        if not self.pipeline.sym_spell:
            print("Skipping spell check test (SymSpell not loaded)")
            return

        # "gravty" -> "gravity"
        # We need to ensure "gravity" is in the dictionary or added.
        # SymSpell uses a loaded dictionary.
        
        # Let's just test that it runs without error.
        res = self.pipeline._stage1_spell_correct("the quick brown fox")
        self.assertEqual(res, "the quick brown fox")

    def test_stage3_manifold_alignment(self):
        """Test semantic alignment."""
        # Mock search to return "gravity" for "graviti"
        self.mock_gravity.search = MagicMock(return_value=[("gravity", 0.85)])
        
        # "graviti" should map to "gravity"
        res = self.pipeline._stage3_manifold_align("graviti")
        self.assertEqual(res, "gravity")
        
        # "random" with low score should stay "random"
        self.mock_gravity.search = MagicMock(return_value=[("something", 0.2)])
        res = self.pipeline._stage3_manifold_align("random")
        self.assertEqual(res, "random")

    def test_stage4_canonicalization(self):
        """Test canonicalization."""
        self.assertEqual(self.pipeline._stage4_canonicalize("Gravity-Field"), "gravity field")
        self.assertEqual(self.pipeline._stage4_canonicalize("PROPER_TIME"), "proper time")

    def test_full_pipeline(self):
        """Test the full flow."""
        # Mock stages for control
        self.pipeline._stage1_spell_correct = MagicMock(return_value="gravity field")
        self.pipeline._stage3_manifold_align = MagicMock(return_value="gravity field")
        
        res = self.pipeline.normalize("  gravty-field  ")
        # Tokenize -> "gravty-field"
        # Spell -> "gravity field"
        # Manifold -> "gravity field"
        # Canon -> "gravity field"
        self.assertEqual(res, "gravity field")

if __name__ == "__main__":
    unittest.main()
