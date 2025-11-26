"""Image-to-image retrieval demo using holographic memory."""

from __future__ import annotations

from pathlib import Path

from hologram import Hologram

DATA_DIR = Path(__file__).resolve().parent / "data"


def _ensure_samples() -> tuple[Path, Path]:
    cat = DATA_DIR / "cat.png"
    dog = DATA_DIR / "dog.png"
    missing = [p for p in (cat, dog) if not p.exists()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise SystemExit(
            f"Missing sample images: {joined}. "
            "Add your own PNGs or rerun the repo setup steps."
        )
    return cat, dog


def _load_hologram() -> Hologram:
    try:
        return Hologram.init()
    except RuntimeError as exc:
        print(f"[warn] {exc}")
        print("[warn] Falling back to hashing-based encoders.")
        return Hologram.init(use_clip=False)


def main() -> None:
    cat, dog = _ensure_samples()
    holo = _load_hologram()

    holo.glyphs.create("image:cat", title="Sample cat image")
    holo.glyphs.create("image:dog", title="Sample dog image")

    holo.add_image_path("image:cat", str(cat))
    holo.add_image_path("image:dog", str(dog))

    print("\n=== Image → Image: using cat sample ===")
    hits = holo.search_image_path(str(cat), top_k=5)
    for trace, score in hits:
        print(holo.summarize_hit(trace, score))

    print("\n=== Image → Image: using dog sample ===")
    hits = holo.search_image_path(str(dog), top_k=5)
    for trace, score in hits:
        print(holo.summarize_hit(trace, score))


if __name__ == "__main__":  # pragma: no cover - manual demo
    main()
