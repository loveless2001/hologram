"""
Extractor logic to transform AST nodes into Hologram Concepts.
"""

from typing import List, Dict, Any

class SymbolExtractor:
    """
    Transforms raw AST symbols into normalized concept definitions.
    """

    def extract(self, raw_nodes: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        concepts = []

        for node in raw_nodes:
            # Construct qualified name
            # e.g., "Spacecraft.launch" or "util.helper"
            # We assume filename might imply a module path, but for now we stick to symbol path
            
            # Combine parents + name
            # parents = ["Spacecraft"], name = "launch" -> "Spacecraft.launch"
            full_path_parts = node["parents"] + [node["name"]]
            qualified_name = ".".join(full_path_parts)
            
            # Create concept ID
            # "concept:<filename>:<qualified_name>"
            # This ensures that symbols with the same name in different files start as separate concepts.
            # Gravity will determine if they should fuse (but we will add protections against that for code).
            
            # Sanitize filename for ID (e.g. replace / with _)
            safe_filename = filename.replace("/", "_").replace("\\", "_").replace(".", "_")
            concept_id = f"concept:{safe_filename}:{qualified_name}"
            
            concepts.append({
                "id": concept_id,
                "name": node["name"],
                "file": filename,
                "span": node["span"],
                "kind": node["type"],
                "doc": node["doc"],
                "parents": node["parents"],
                "qualified_name": qualified_name
            })
            
        return concepts
