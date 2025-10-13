# api_server/main.py
from fastapi import FastAPI
from hologram.api import Hologram

app = FastAPI()
memory = Hologram()

@app.post("/add_text")
def add_text(glyph_id: str, text: str):
    return {"trace_id": memory.add_text(glyph_id, text)}

@app.get("/search")
def search(q: str, top_k: int = 5):
    hits = memory.search_text(q, top_k)
    return [{"trace": t.content, "score": s} for t, s in hits]
