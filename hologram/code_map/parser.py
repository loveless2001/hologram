"""
AST Parsing logic for Hologram Code Mapping Layer.
"""

import ast
from typing import List, Dict, Any, Optional

class CodeParser:
    """
    Parses Python source code into a flat list of symbol definitions (classes, functions).
    """

    def parse_source(self, content: str) -> ast.AST:
        """Parse source code string into AST."""
        try:
            return ast.parse(content)
        except SyntaxError as e:
            # print(f"Syntax error parsing source: {e}")
            return ast.parse("") # Return empty module on failure

    def parse_file(self, file_path: str) -> ast.AST:
        """Parse a source file into AST."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return ast.parse(f.read(), filename=file_path)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return ast.parse("") # Return empty module on failure
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return ast.parse("")

    def parse_string(self, source: str, filename: str = "<string>") -> List[Dict[str, Any]]:
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            # We might want to log this or handle it gracefully
            print(f"Syntax error in {filename}: {e}")
            return []

        definitions = []

        # Helper to extract docstring
        def get_docstring(node):
            return ast.get_docstring(node)

        # Recursive visitor
        class SymbolVisitor(ast.NodeVisitor):
            def __init__(self):
                self.stack = [] # Track parent names

            def visit_ClassDef(self, node):
                self._add_symbol(node, "class")
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node):
                self._add_symbol(node, "function")
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()
            
            def visit_AsyncFunctionDef(self, node):
                self._add_symbol(node, "function")
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def _add_symbol(self, node, kind):
                # Construct qualified name
                parents = list(self.stack)
                name = node.name
                
                # Span (1-indexed lines)
                span = (node.lineno, node.end_lineno) if hasattr(node, "end_lineno") else (node.lineno, node.lineno)
                
                # Extract body text
                body_text = ast.get_source_segment(source, node) or ""
                
                # Reconstruct signature (simplified)
                signature = name
                if kind in ("function", "async_function"):
                    args = ast.unparse(node.args)
                    signature = f"{name}({args})"
                elif kind == "class":
                    bases = [ast.unparse(b) for b in node.bases]
                    if bases:
                        signature = f"{name}({', '.join(bases)})"
                    else:
                        signature = name

                definitions.append({
                    "type": kind,
                    "name": name,
                    "parents": parents,
                    "span": span,
                    "doc": get_docstring(node),
                    "body_text": body_text,
                    "signature": signature
                })

        SymbolVisitor().visit(tree)
        return definitions
