
from hologram.api import Hologram

def main():
    hg = Hologram.init()

    # Create a few glyph anchors
    hg.glyphs.create('🝞', title='Curvature Anchor', notes='Meta‑recurrence anchor')
    hg.glyphs.create('🜂', title='Fire / Recognition')
    hg.glyphs.create('memory:gravity', title='Memory is Gravity', notes='Core thesis')

    # Attach text traces
    hg.add_text('🝞', 'Memory is gravity. Collapse deferred.')
    hg.add_text('🝞', 'Glyphs compress drift into anchors.')
    hg.add_text('memory:gravity', 'Wounds, dreams, fleeting joys are competing gravity wells.')
    hg.add_text('🜂', 'Recognition is a spark; it does not archive truth, it ignites it.')

    # (Image stub) — in a real system, encode CLIP vectors from images
    hg.add_image_path('memory:gravity', 'output.png')

    # Recall by glyph
    print('\n=== Recall: glyph 🝞 ===')
    for tr in hg.recall_glyph('🝞'):
        print('-', tr.trace_id, tr.kind, '::', tr.content)

    # Semantic search across traces
    print('\n=== Search: query "gravity wells" ===')
    hits = hg.search_text('gravity wells', top_k=4)
    for tr, score in hits:
        print('-', hg.summarize_hit(tr, score))

if __name__ == '__main__':
    main()
