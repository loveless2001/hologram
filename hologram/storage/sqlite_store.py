
import sqlite3
import json
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from ..store import Trace, Glyph

class SqliteBackend:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    kind TEXT,
                    content TEXT,
                    vec BLOB,
                    meta TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS glyphs (
                    glyph_id TEXT PRIMARY KEY,
                    title TEXT,
                    notes TEXT,
                    meta TEXT
                )
            """)
            # Try to add meta column to glyphs if it doesn't exist (for migration)
            try:
                self.conn.execute("ALTER TABLE glyphs ADD COLUMN meta TEXT")
            except sqlite3.OperationalError:
                pass # Already exists

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS glyph_traces (
                    glyph_id TEXT,
                    trace_id TEXT,
                    PRIMARY KEY (glyph_id, trace_id),
                    FOREIGN KEY(glyph_id) REFERENCES glyphs(glyph_id),
                    FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    def save_meta(self, key: str, value: str):
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def load_meta(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def save_trace(self, trace: Trace):
        vec_bytes = trace.vec.astype(np.float32).tobytes()
        
        # Pack extra fields into meta for SQL storage
        # We assume trace.meta is a dict.
        storage_meta = trace.meta.copy()
        if trace.span:
            storage_meta["_span"] = trace.span
        if trace.source_file:
            storage_meta["_source_file"] = trace.source_file
        if trace.resolved_text:
            storage_meta["_resolved_text"] = trace.resolved_text
        if trace.coref_map:
            storage_meta["_coref_map"] = trace.coref_map
            
        meta_json = json.dumps(storage_meta)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, kind, content, vec, meta) VALUES (?, ?, ?, ?, ?)",
                (trace.trace_id, trace.kind, trace.content, vec_bytes, meta_json)
            )

    def load_traces(self) -> List[Trace]:
        traces = []
        cur = self.conn.execute("SELECT trace_id, kind, content, vec, meta FROM traces")
        for row in cur:
            trace_id, kind, content, vec_blob, meta_json = row
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            storage_meta = json.loads(meta_json)
            
            # Unpack extra fields
            span = tuple(storage_meta.pop("_span")) if "_span" in storage_meta else None
            source_file = storage_meta.pop("_source_file", None)
            resolved_text = storage_meta.pop("_resolved_text", None)
            coref_map = storage_meta.pop("_coref_map", None)
            
            traces.append(Trace(
                trace_id=trace_id, 
                kind=kind, 
                content=content, 
                vec=vec, 
                meta=storage_meta,
                span=span,
                source_file=source_file,
                resolved_text=resolved_text,
                coref_map=coref_map
            ))
        return traces

    def save_glyph(self, glyph: Glyph):
        glyph_meta = json.dumps(glyph.code_meta) if glyph.code_meta else None
        
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO glyphs (glyph_id, title, notes, meta) VALUES (?, ?, ?, ?)",
                (glyph.glyph_id, glyph.title, glyph.notes, glyph_meta)
            )
            for tid in glyph.trace_ids:
                self.conn.execute(
                    "INSERT OR IGNORE INTO glyph_traces (glyph_id, trace_id) VALUES (?, ?)",
                    (glyph.glyph_id, tid)
                )

    def load_glyphs(self) -> List[Glyph]:
        glyphs = {}
        # Try to select meta, handle if older schema (though _init_schema tries to fix it)
        # But for robustness, we can just select * or specifically try.
        # Given _init_schema runs on init, we can assume 'meta' column exists.
        cur = self.conn.execute("SELECT glyph_id, title, notes, meta FROM glyphs")
        for row in cur:
            glyph_id, title, notes, meta_json = row
            code_meta = json.loads(meta_json) if meta_json else None
            glyphs[glyph_id] = Glyph(glyph_id, title, notes, trace_ids=[], code_meta=code_meta)
        
        cur = self.conn.execute("SELECT glyph_id, trace_id FROM glyph_traces")
        for row in cur:
            glyph_id, trace_id = row
            if glyph_id in glyphs:
                glyphs[glyph_id].trace_ids.append(trace_id)
                
        return list(glyphs.values())
        
    def close(self):
        self.conn.close()
