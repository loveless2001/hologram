from hologram.text_utils import extractor
import logging

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

sentences = [
    "Time dilation occurs near the speed of light.",
    "Mass-energy equivalence states that energy equals mass.",
    "The baseball player hit the ball with a bat.",
    "Gravity attracts massive objects."
]

print("--- Current Labels ---")
print(f"Labels: {extractor.labels}")
for s in sentences:
    print(f"\nInput: {s}")
    print(f"Extracted: {extractor.extract_concepts(s)}")

print("\n--- Experimenting with Relation Labels ---")
# Temporarily add relation-focused labels (deduplicated)
original_labels = extractor.labels.copy()
extra_labels = ["relation", "action", "verb", "predicate", "interaction"]
new_labels = list(set(original_labels + extra_labels))
extractor.labels = new_labels

print(f"New Labels: {extractor.labels}")
for s in sentences:
    print(f"\nInput: {s}")
    print(f"Extracted: {extractor.extract_concepts(s)}")

# Restore
extractor.labels = original_labels
