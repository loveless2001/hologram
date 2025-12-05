"""
Hologram FastAPI Server
Exposes Hologram's 3-tier ontology system via REST API for VSCode extension.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import uvicorn
import os

from hologram.api import Hologram
from hologram.gravity import TIER_DOMAIN, TIER_SYSTEM

app = FastAPI(
    title="Hologram Server",
    description="Semantic code memory with 3-tier ontology",
    version="1.0.0"
)

# CORS for localhost VSCode extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Hologram instances per project
hologram_instances: Dict[str, Hologram] = {}

# Load global configuration if available
from .config import Config
from .global_config import global_project_exists, apply_global_config, SyncStatus, sync_config

# Apply global config on startup (env vars still override)
if global_project_exists():
    status, diff = sync_config()
    if status == SyncStatus.SYNCED:
        print("[Server] Global config loaded (synchronized)")
    elif status == SyncStatus.CONFLICT:
        print(f"[Server] Config conflict detected ({len(diff)} differences)")
        print("[Server] Applying global config (env vars still override)")
        apply_global_config(apply_env_overrides=True)
else:
    print("[Server] No global config found, using defaults")

MEMORY_DIR = Path(Config.storage.MEMORY_DIR)
MEMORY_DIR.mkdir(exist_ok=True)


# ============================================================================
# Request/Response Models
# ============================================================================

class IngestRequest(BaseModel):
    project: str
    text: str
    path: Optional[str] = None
    origin: str = "code"  # code, docs, config, manual
    tier: int = TIER_DOMAIN
    metadata: Optional[Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    project: str
    text: str
    top_k: int = 5


class MemorySummary(BaseModel):
    project: str
    total_concepts: int
    tier1_count: int
    tier2_count: int
    projects: List[str]
    recent_traces: List[Dict[str, Any]]


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_create_hologram(project: str) -> Hologram:
    """Get existing Hologram instance or create new one."""
    if project not in hologram_instances:
        # Determine file extension based on config
        file_ext = Config.storage.SQLITE_DB_NAME if Config.storage.USE_SQLITE else "memory.json"
        project_dir = MEMORY_DIR / project
        project_dir.mkdir(parents=True, exist_ok=True)
        save_path = project_dir / file_ext
        
        # Also check for legacy .json file if SQLite is enabled
        legacy_json_path = project_dir / "memory.json"
        
        if save_path.exists():
            print(f"[Server] Loading existing memory for project '{project}' from {file_ext}")
            hologram_instances[project] = Hologram.load(
                str(save_path),
                encoder_mode="minilm",
                use_gravity=True,
                auto_ingest_system=True
            )
        elif Config.storage.USE_SQLITE and legacy_json_path.exists():
            # Migrate from JSON to SQLite
            print(f"[Server] Migrating project '{project}' from JSON to SQLite")
            hologram_instances[project] = Hologram.load(
                str(legacy_json_path),
                encoder_mode="minilm",
                use_gravity=True,
                auto_ingest_system=True
            )
            # Save in new format
            hologram_instances[project].save(str(save_path))
            print(f"[Server] Migration complete, saved to {save_path}")
        else:
            print(f"[Server] Creating new Hologram for project '{project}'")
            hologram_instances[project] = Hologram.init(
                encoder_mode="minilm",
                use_gravity=True,
                auto_ingest_system=True
            )
        
        hologram_instances[project].project = project
    
    return hologram_instances[project]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Hologram Server",
        "version": "1.0.0",
        "active_projects": list(hologram_instances.keys())
    }


@app.post("/ingest")
async def ingest_text(req: IngestRequest):
    """
    Ingest text into project-specific Hologram instance.
    
    Supports tier-aware ingestion with automatic concept extraction.
    """
    try:
        holo = get_or_create_hologram(req.project)
        
        # Create glyph ID from path or hash
        if req.path:
            glyph_id = f"file:{req.path}"
        else:
            glyph_id = f"text:{abs(hash(req.text)) % 10**10}"
        
        # Add to hologram with tier awareness
        trace_id = holo.add_text(
            glyph_id=glyph_id,
            text=req.text,
            tier=req.tier,
            origin=req.origin,
            **req.metadata
        )
        
        return {
            "status": "success",
            "trace_id": trace_id,
            "project": req.project,
            "tier": req.tier,
            "origin": req.origin
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_memory(req: QueryRequest):
    """
    Query project memory using probe physics.
    
    Returns relevant memory nodes, edges, and trajectory information.
    """
    if req.project not in hologram_instances:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{req.project}' not found. Ingest some data first."
        )
    
    try:
        holo = hologram_instances[req.project]
        
        # Use retrieve method for probe-based search
        packet = holo.retrieve(req.text)
        
        return {
            "query": req.text,
            "nodes": packet.nodes[:req.top_k],
            "edges": packet.edges[:10],  # Limit edges
            "glyphs": packet.glyphs[:5],
            "trajectory_steps": packet.trajectory_steps
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{project}")
async def get_memory_summary(project: str):
    """
    Get memory summary for project.
    
    Returns concept counts by tier, recent traces, and project metadata.
    """
    if project not in hologram_instances:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project}' not found"
        )
    
    try:
        holo = hologram_instances[project]
        
        # Count concepts by tier (exclude aliases)
        tier1 = sum(
            1 for c in holo.field.sim.concepts.values()
            if c.tier == TIER_DOMAIN and c.canonical_id is None
        )
        tier2 = sum(
            1 for c in holo.field.sim.concepts.values()
            if c.tier == TIER_SYSTEM and c.canonical_id is None
        )
        
        # Get recent traces
        recent = list(holo.store.traces.values())[-10:]
        recent_data = [
            {
                "id": t.trace_id,
                "kind": t.kind,
                "content": t.content[:100] + "..." if len(t.content) > 100 else t.content,
                "meta": t.meta
            }
            for t in recent
        ]
        
        # Get unique projects
        projects = list(set(
            c.project for c in holo.field.sim.concepts.values()
            if c.canonical_id is None
        ))
        
        return {
            "project": project,
            "total_concepts": len([
                c for c in holo.field.sim.concepts.values()
                if c.canonical_id is None
            ]),
            "tier1_count": tier1,
            "tier2_count": tier2,
            "projects": projects,
            "recent_traces": recent_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save/{project}")
async def save_memory(project: str, path: Optional[str] = None):
    """
    Save project memory to disk.
    
    Defaults to ~/.hologram_memory/<project>/memory.db (SQLite) or memory.json
    """
    if project not in hologram_instances:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project}' not found"
        )
    
    try:
        if path is None:
            project_dir = MEMORY_DIR / project
            project_dir.mkdir(exist_ok=True)
            # Use config to determine file format
            file_name = Config.storage.SQLITE_DB_NAME if Config.storage.USE_SQLITE else "memory.json"
            path = str(project_dir / file_name)
        
        hologram_instances[project].save(path)
        
        return {
            "status": "saved",
            "project": project,
            "path": path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load/{project}")
async def load_memory(project: str, path: str):
    """
    Load project memory from disk.
    
    Creates new Hologram instance from saved state.
    """
    if not Path(path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Memory file not found: {path}"
        )
    
    try:
        hologram_instances[project] = Hologram.load(
            path,
            encoder_mode="minilm",
            use_gravity=True,
            auto_ingest_system=True
        )
        hologram_instances[project].project = project
        
        return {
            "status": "loaded",
            "project": project,
            "path": path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects")
async def list_projects():
    """List all active projects."""
    return {
        "projects": list(hologram_instances.keys()),
        "count": len(hologram_instances)
    }


@app.delete("/project/{project}")
async def delete_project(project: str):
    """Remove project from memory (does not delete saved files)."""
    if project in hologram_instances:
        del hologram_instances[project]
        return {"status": "deleted", "project": project}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project}' not found"
        )


# ============================================================================
# Server Startup
# ============================================================================

def start_server(host: str = None, port: int = None, reload: bool = None):
    """Start the Hologram server."""
    host = host or Config.server.HOST
    port = port or Config.server.PORT
    if reload is None: reload = Config.server.RELOAD
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║              🌀 Hologram Server Starting                 ║
╠═══════════════════════════════════════════════════════════╣
║  Host: {host:<48} ║
║  Port: {port:<48} ║
║  Memory Dir: {str(MEMORY_DIR):<42} ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(
        "hologram.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_server(reload=True)
