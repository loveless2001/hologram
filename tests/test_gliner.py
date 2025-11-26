from hologram.text_utils import extract_concepts
import logging

logging.basicConfig(level=logging.INFO)

text = "Special Relativity describes how time dilation occurs near the speed of light."
print(f"Input: {text}")
concepts = extract_concepts(text)
print(f"Concepts: {concepts}")

text2 = "Apple pie is a popular dessert in America."
print(f"Input: {text2}")
concepts2 = extract_concepts(text2)
print(f"Concepts: {concepts2}")
