
import unittest
import numpy as np
from unittest.mock import MagicMock
from hologram.text_utils import clean_text, correct_spelling, normalize_text, sym_spell

class TestRobustness(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text("foo—bar"), "foo-bar")
        self.assertEqual(clean_text("foo…"), "foo...")

    def test_correct_spelling(self):
        if sym_spell is None:
            print("SymSpell not loaded, skipping spelling test")
            return
        # "gravty" -> "gravity"
        self.assertEqual(correct_spelling("gravty"), "gravity")
        # "dilaation" -> "dilation"
        self.assertEqual(correct_spelling("dilaation"), "dilation")
        # "hello world" -> "hello world"
        self.assertEqual(correct_spelling("hello world"), "hello world")

    def test_normalize_text_fuzzy(self):
        # Mock store and encoder
        mock_store = MagicMock()
        mock_encoder = MagicMock()
        
        # Setup mock encoder
        mock_encoder.encode.return_value = np.array([0.1, 0.2])
        
        # Setup mock store search
        # Case 1: High similarity -> resolve to canonical
        # search_traces returns [(trace_id, score)]
        mock_store.search_traces.return_value = [("trace_123", 0.9)]
        
        # Setup mock store get_trace
        mock_trace = MagicMock()
        mock_trace.content = "canonical concept"
        mock_store.get_trace.return_value = mock_trace
        
        # Input: "canonnical concpt" (misspelled)
        # It should first be spell corrected (if possible) or just cleaned.
        # Let's assume spell corrector fixes it or not.
        # If we pass a string that spell corrector doesn't change much, but fuzzy resolver finds match.
        
        # normalize_text calls clean -> correct -> fuzzy
        
        # Let's test fuzzy logic specifically.
        # We pass a string that is already "clean" but maybe not canonical.
        input_text = "canonical concept variant" 
        # If we want to force fuzzy match, we need search_traces to return high score for this input's vector.
        
        result = normalize_text(input_text, store=mock_store, encoder=mock_encoder)
        
        # Expectation: It should return "canonical concept" because score 0.9 > 0.75
        self.assertEqual(result, "canonical concept")
        
        # Case 2: Low similarity -> keep original
        mock_store.search_traces.return_value = [("trace_456", 0.5)]
        result = normalize_text("completely new concept", store=mock_store, encoder=mock_encoder)
        self.assertEqual(result, "completely new concept")

    def test_normalize_text_ambiguity(self):
        # Mock store and encoder
        mock_store = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([0.1, 0.2])
        
        # Case 3: Gray zone (0.60 < score < 0.75) -> Log it
        mock_store.search_traces.return_value = [("trace_789", 0.70)]
        mock_trace = MagicMock()
        mock_trace.content = "ambiguous concept"
        mock_store.get_trace.return_value = mock_trace
        
        # We want to verify that _log_ambiguity is called.
        with unittest.mock.patch('hologram.text_utils._log_ambiguity') as mock_log:
            # "ambigous" (typo) -> "ambigous" (returned) but logged against "ambiguous concept"
            # Note: spell corrector might fix "ambigous" -> "ambiguous" if it's in dict.
            # Let's use a word that definitely isn't in dict or is just a variant.
            # "Quantum Fields" vs "Quantum Field Theory"
            
            input_text = "Quantum Fields"
            mock_trace.content = "Quantum Field Theory"
            
            result = normalize_text(input_text, store=mock_store, encoder=mock_encoder)
            
            # SymSpell likely lowercases "Quantum" -> "quantum"
            # So we expect the result to be lowercased if it went through correct_spelling
            # "Quantum Fields" -> "quantum fields"
            expected_result = "quantum fields"
            
            # Should return original text (passive acceptance) - but spell corrected
            self.assertEqual(result, expected_result)
            
            # Should call log
            mock_log.assert_called_once()
            args, _ = mock_log.call_args
            self.assertEqual(args[0], expected_result)
            self.assertEqual(args[1], "Quantum Field Theory")
            self.assertAlmostEqual(args[2], 0.70)


if __name__ == '__main__':
    unittest.main()
