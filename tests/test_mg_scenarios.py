import pytest
import numpy as np
from hologram.api import Hologram

# Test data from mgscore_testing.md

SCENARIOS = {
    "1.1_perfect_coherence": {
        "texts": [
            "The cat is sleeping on the couch.",
            "The cat is resting on the sofa.",
            "The cat lies quietly on the couch.",
            "The cat is napping on the sofa."
        ],
        "checks": lambda s: s.coherence > 0.7 and s.collapse_risk < 0.15
    },
    "2.1_random_jumps": {
        "texts": [
            "Bananas contain potassium.",
            "Quantum entanglement challenges classical locality.",
            "My shoes got wet yesterday.",
            "The stock market reacts unpredictably to geopolitical tension."
        ],
        "checks": lambda s: s.coherence < 0.6 and s.collapse_risk > 0.05
    },
    "3.1_linear_argument": {
        "texts": [
            "Climate change leads to more extreme weather events.",
            "Extreme weather events increase infrastructure damage.",
            "Infrastructure damage raises national recovery costs.",
            "Higher recovery costs strain government budgets."
        ],
        "checks": lambda s: s.curvature > 0.65  # Linear-ish (~0.7)
    },
    "3.2_sharp_turn": {
        "texts": [
            "Machine learning models improve by analyzing patterns in data.",
            "Deep neural networks are especially good at extracting nonlinear features.",
            "However, medieval architecture relied heavily on flying buttresses to support cathedral roofs."
        ],
        "checks": lambda s: s.curvature < 0.65  # Sharp turn (~0.55)
    }
}

@pytest.fixture(scope="module")
def hologram():
    # Use minilm for better semantic quality if available, else hash
    try:
        return Hologram.init(encoder_mode="minilm", use_gravity=False)
    except Exception:
        return Hologram.init(encoder_mode="hash", use_gravity=False)

@pytest.mark.parametrize("scenario_name", SCENARIOS.keys())
def test_scenario(hologram, scenario_name):
    data = SCENARIOS[scenario_name]
    texts = data["texts"]
    check_fn = data["checks"]
    
    score = hologram.score_text(texts)
    
    print(f"\nScenario: {scenario_name}")
    print(f"  Coherence: {score.coherence:.4f}")
    print(f"  Entropy:   {score.entropy:.4f}")
    print(f"  Curvature: {score.curvature:.4f}")
    print(f"  Risk:      {score.collapse_risk:.4f}")
    
    assert check_fn(score), f"Failed checks for {scenario_name}"
