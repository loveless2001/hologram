# api_server/main.py
import os
import shutil
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles  # Added for viz
from pydantic import BaseModel

from hologram.api import Hologram
from hologram.chatbot import ChatMemory, resolve_provider
from hologram.text_utils import extract_concepts

app = FastAPI()

# Mount static files for visualization
# Ensure directory exists
static_dir = Path("api_server/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/viz", StaticFiles(directory="api_server/static", html=True), name="viz")

# Global state - PROTECTED BY LOCK
KB_DIR = Path("data/kbs")
KB_DIR.mkdir(parents=True, exist_ok=True)
current_kb_name: Optional[str] = None
memory: Optional[Hologram] = None
chat_memory: Optional[ChatMemory] = None
_kb_lock = threading.Lock()  # Protects all global state

# Internal helper - assumes lock is already held
def _load_kb_internal(kb_name: Optional[str]):
    global memory, chat_memory, current_kb_name
    
    if kb_name is None:
        memory = None
        chat_memory = None
        current_kb_name = None
        return

    # Check for JSON dump (full state)
    if kb_name.endswith(".json"):
        # Try root first, then KB_DIR
        path = Path(kb_name)
        if not path.exists():
            path = KB_DIR / kb_name
        
        if path.exists():
            print(f"Loading KB from JSON dump: {path}")
            # Load with gravity enabled to support visualization
            memory = Hologram.load(str(path), use_clip=False, use_gravity=True)
            current_kb_name = kb_name
            chat_memory = ChatMemory(memory)
            return

    # Default: Load from TXT (rebuild)
    kb_path = KB_DIR / kb_name
    if not kb_path.exists():
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_name}")

    # Initialize new memory
    memory = Hologram.init(use_gravity=True)
    
    # Load content line by line
    print(f"Loading and decomposing KB: {kb_name}")
    with open(kb_path, "r") as f:
        for line in f:
            text = line.strip()
            if text:
                # Decompose text into atomic concepts
                concepts = extract_concepts(text)
                for concept in concepts:
                    # Add each concept to the root glyph
                    memory.add_text("root", concept)
    
    current_kb_name = kb_name
    chat_memory = ChatMemory(memory)

# Public API - acquires lock
def load_kb(kb_name: Optional[str]):
    with _kb_lock:
        _load_kb_internal(kb_name)

# Initial load (empty)
load_kb(None)

class ChatRequest(BaseModel):
    message: str
    kb_name: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class ConceptRelation(BaseModel):
    concept: str
    strength: float

class SearchResult(BaseModel):
    content: str
    score: float
    relations: Optional[List[ConceptRelation]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.get("/")
def health_check():
    return {"status": "ok", "service": "hologram-api"}
    
@app.get("/kbs")
def list_kbs():
    return {"kbs": [f.name for f in KB_DIR.glob("*.txt")]}

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """Search for semantic matches with concept relations."""
    with _kb_lock:
        if not memory:
            raise HTTPException(status_code=400, detail="No KB loaded. Load a KB first.")
        
        # Perform semantic search
        hits = memory.search_text(request.query, top_k=request.top_k)
        
        results = []
        for trace, score in hits:
            # Get relations for this concept from gravity field
            relations_list = []
            if memory.field and hasattr(memory.field, 'sim'):
                gravity = memory.field.sim
                trace_id = trace.trace_id
                
                # Find relations for this trace_id in the gravity field
                if trace_id in gravity.concepts:
                    # Get all relations involving this concept
                    for (c1, c2), strength in gravity.relations.items():
                        if c1 == trace_id or c2 == trace_id:
                            # Get the other concept in the relation
                            other_id = c2 if c1 == trace_id else c1
                            
                            # Resolve to content
                            if other_id in memory.store.traces:
                                other_content = memory.store.traces[other_id].content
                                relations_list.append(
                                    ConceptRelation(
                                        concept=other_content,
                                        strength=float(strength)
                                    )
                                )
                    
                    # Sort by strength and take top 5
                    relations_list.sort(key=lambda r: abs(r.strength), reverse=True)
                    relations_list = relations_list[:5]
            
            results.append(
                SearchResult(
                    content=trace.content,
                    score=float(score),
                    relations=relations_list if relations_list else None
                )
            )
        
        return SearchResponse(query=request.query, results=results)

@app.get("/viz-data")
def get_viz_data():
    """Return 2D projection of the current memory field."""
    with _kb_lock:
        if not memory or not memory.field:
            return {"points": [], "labels": []}
        
        proj, ids = memory.field_state()
        
        # Resolve IDs to content
        labels = []
        for tid in ids:
            if memory.store and tid in memory.store.traces:
                # Use content from trace, truncated if too long
                content = memory.store.traces[tid].content
                labels.append(content[:50] + "..." if len(content) > 50 else content)
            else:
                # Fallback to ID
                labels.append(tid)

        # Convert numpy array to list of lists for JSON serialization
        return {
            "points": proj.tolist(),
            "labels": labels
        }

@app.post("/kbs/upload")
async def upload_kb(file: UploadFile = File(...)):
    file_path = KB_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.delete("/kbs/{name}")
def delete_kb(name: str):
    file_path = KB_DIR / name
    if file_path.exists():
        file_path.unlink()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="KB not found")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    with _kb_lock:  # Protect global state access
        global current_kb_name
        
        # Switch KB if requested and different
        if request.kb_name and request.kb_name != current_kb_name:
            _load_kb_internal(request.kb_name)  # Use internal helper to avoid deadlock
        
        # Use a simple provider for now (Echo or OpenAI if key set)
        provider = resolve_provider()
        
        # Construct context from memory
        # We'll manually do what ChatWindow.step does but stateless-ish
        session_id = "web_session" # Single session for now for simplicity
        
        # 1. Search for context
        context_msgs = chat_memory.get_recent_session_messages(session_id)
        cross_prompt = chat_memory.build_cross_session_prompt(session_id, request.message)
        
        # 2. Build messages
        messages = []
        if cross_prompt:
            messages.append({"role": "system", "content": cross_prompt})
        messages.extend(context_msgs)
        messages.append({"role": "user", "content": request.message})
        
        # 3. Generate reply
        reply = provider.generate(messages)
        
        # 4. Record interaction
        chat_memory.record_message(session_id, "user", request.message)
        chat_memory.record_message(session_id, "assistant", reply)
        
        return {"reply": reply}

