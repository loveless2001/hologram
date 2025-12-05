"""
Mapper logic to convert Concepts into Glyphs and Traces.
"""

from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
from ..store import Glyph, Trace
import uuid

class ConceptMapper:
    """
    Maps normalized concept dictionaries to Hologram Glyphs and Traces.
    """

    def map_concept(
        self, 
        concept: Dict[str, Any], 
        vectorizer: Callable[[str], np.ndarray]
    ) -> Tuple[Glyph, Trace]:
        """
        Creates a Glyph and a Trace for a given concept.
        """
        
        # 1. Construct text representation for embedding (The Trace content)
        # We want the vector to Capture the *meaning* of the code.
        # "Function abstract_mission used for abort sequences..."
        
        doc = concept.get("doc") or ""
        kind = concept.get("kind")
        name = concept.get("name")
        qname = concept.get("qualified_name")
        filename = concept.get("file")
        
        # Content for embedding
        content_text = f"{kind.capitalize()} {qname} defined in {filename}.\n"
        if doc:
            content_text += f"Docstring: {doc}\n"
        
        # We might also want to include the source code snippet itself if small enough?
        # For now, just docstring and signature context.
        
        # Generate Vector
        vec = vectorizer(content_text)
        
        # 2. Create Trace
        # Trace ID: unique ID for this specific occurrence/definition
        # "trace:code:<file>:<span_start>"
        span_str = f"{concept['span'][0]}-{concept['span'][1]}"
        trace_id = f"trace:code:{filename}:{qname}" 
        # Note: filenames might have slashes, making ID messy? 
        # Usually IDs are opaque or hash-based.
        # Let's use a hash or UUID to be safe, but readable is nice.
        # Let's stick to a structured string but maybe sanitized.
        # For now, simple string.
        
        trace = Trace(
            trace_id=trace_id,
            kind="code",
            content=content_text,
            vec=vec,
            meta={
                "parents": concept["parents"],
                "code_type": kind
            },
            span=concept["span"],
            source_file=filename
        )
        # Update Trace model later to explicitly have source_file/span fields 
        # or just rely on meta for now (though plan says "Extend Trace...").
        # I'll put them in meta first, and `api.py` or `store.py` updates will make them first-class if needed.
        # The plan says "Modify Trace... source_file: Optional[str]".
        # So I will assume the `Trace` constructor will eventually accept these or I set them.
        # Python's dataclass constructor needs exact fields. 
        # I will rely on `api.py` to handle the modified Trace class, 
        # OR I should check if I can modify Trace class first.
        # I'll stick to `meta` for safety in this file, then update `store.py` to add fields,
        # then update this file to use them if I can.
        # Actually, I am supposed to modify `store.py` in Phase 2.
        # So safely, I can use kwargs or just set attributes after init if dataclass allows.
        # But `meta` is safe.
        
        # 3. Create Glyph
        # Glyph ID: The concept itself. 
        glyph_id = concept["id"]
        
        glyph = Glyph(
            glyph_id=glyph_id,
            title=qname,
            notes=f"Code symbol from {filename}",
            trace_ids=[trace_id]
        )
        
        return glyph, trace
