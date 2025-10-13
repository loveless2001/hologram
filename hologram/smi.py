from tinydb import TinyDB

class SMI:
    def __init__(self, path="data/smi.json"):
        self.db = TinyDB(path)

    def log_anchor(self, glyph_id, resonance):
        self.db.insert({
            "id": glyph_id,
            "resonance": resonance,
            "timestamp": time.time(),
        })

    def decay(self):
        for row in self.db:
            row["resonance"] *= 0.98  # exponential decay
            self.db.update(row)
    
self.smi.log_anchor(glyph_id, resonance)
