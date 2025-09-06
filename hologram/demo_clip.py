# demo_clip.py
from hologram.api import Hologram

def main():
    hg = Hologram.init()

    hg.glyphs.create("images:animals", title="Animal images")

    # Add two images (make sure these exist on disk)
    hg.add_image_path("images:animals", "./data/cat.jpg")
    hg.add_image_path("images:animals", "./data/dog.jpg")

    # Query with text
    print("\n=== Search: 'a photo of a cat' ===")
    hits = hg.search_text("a photo of a cat", top_k=5)
    for tr, score in hits:
        print(hg.summarize_hit(tr, score))

    print("\n=== Search: 'a photo of a dog' ===")
    hits = hg.search_text("a photo of a dog", top_k=5)
    for tr, score in hits:
        print(hg.summarize_hit(tr, score))

if __name__ == "__main__":
    main()
