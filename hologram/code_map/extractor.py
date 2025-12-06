"""
Extractor logic to transform AST nodes into Hologram Concepts.
"""

from typing import List, Dict, Any
import hashlib

class SymbolExtractor:
    """
    Transforms raw AST symbols into normalized concept definitions.
    """

    def compute_symbol_id(self, language: str, name: str, signature: str) -> str:
        """
        Deterministic ID based on symbol semantics, not location.
        Allows moving functions between files without breaking identity.
        ID = sha256(language + normalized_name + normalized_signature)
        """
        # Normalize inputs to ensure stability
        # Remove whitespace from signature to be robust against formatting changes
        norm_name = name.strip()
        norm_sig = "".join(signature.split()) # Remove all whitespace
        
        payload = f"{language}:{norm_name}:{norm_sig}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def extract(self, raw_nodes: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        concepts = []

        for node in raw_nodes:
            # Construct qualified name
            # e.g., "Spacecraft.launch" or "util.helper"
            full_path_parts = node["parents"] + [node["name"]]
            qualified_name = ".".join(full_path_parts)
            
            # Determine signature (fallback if not present in node, though it should be)
            # In Phase 1 parser, we might need to assume 'signature' is available or construct it.
            # Ideally the parser provides it. If not, we use qualified_name as proxy for now.
            signature = node.get("signature", qualified_name)
            
            # Compute determinstic ID
            # Assuming Python for now
            symbol_id = self.compute_symbol_id("python", qualified_name, signature)
            
            concepts.append({
                "id": symbol_id, # Pure hash ID
                "name": node["name"],
                "file": filename,
                "span": node["span"],
                "kind": node["type"],
                "doc": node["doc"],
                "parents": node["parents"],
                "qualified_name": qualified_name,
                "signature": signature,
                "body_text": node.get("body_text", "") # Ensure body text is passed for embedding
            })
            
        return concepts
