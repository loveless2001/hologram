
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
                    notes TEXT
                )
            """)
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
        meta_json = json.dumps(trace.meta)
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
            meta = json.loads(meta_json)
            traces.append(Trace(trace_id, kind, content, vec, meta))
        return traces

    def save_glyph(self, glyph: Glyph):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO glyphs (glyph_id, title, notes) VALUES (?, ?, ?)",
                (glyph.glyph_id, glyph.title, glyph.notes)
            )
            # Update links
            # First delete existing links for this glyph? Or merge?
            # Store logic says "upsert_glyph" merges trace_ids.
            # Here we just save what's in the Glyph object.
            # But if we want to be efficient, we should only insert new ones.
            # For simplicity, let's just insert ignore.
            for tid in glyph.trace_ids:
                self.conn.execute(
                    "INSERT OR IGNORE INTO glyph_traces (glyph_id, trace_id) VALUES (?, ?)",
                    (glyph.glyph_id, tid)
                )

    def load_glyphs(self) -> List[Glyph]:
        glyphs = {}
        cur = self.conn.execute("SELECT glyph_id, title, notes FROM glyphs")
        for row in cur:
            glyph_id, title, notes = row
            glyphs[glyph_id] = Glyph(glyph_id, title, notes, trace_ids=[])
        
        # Load links
        cur = self.conn.execute("SELECT glyph_id, trace_id FROM glyph_traces")
        for row in cur:
            glyph_id, trace_id = row
            if glyph_id in glyphs:
                glyphs[glyph_id].trace_ids.append(trace_id)
                
        return list(glyphs.values())
        
    def close(self):
        self.conn.close()
