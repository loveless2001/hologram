#!/usr/bin/env python3
"""
Adaptive Code Analysis Workflow Implementation
Follows .agent/workflows/analyze-code.md phases
"""

import os
import ast
import json
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(".").resolve()
OUTPUT_DIR = PROJECT_ROOT / ".agent" / "memory"
IGNORE_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".vscode", ".agent"}

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = {}
        self.functions = {}
        self.imports = []
        self.globals = []
        self.configs = []
        self.current_class = None

    def visit_ClassDef(self, node):
        self.classes[node.name] = {
            "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
            "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
            "lineno": node.lineno
        }
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        key = f"{self.current_class}.{node.name}" if self.current_class else node.name
        self.functions[key] = {
            "args": [a.arg for a in node.args.args],
            "complexity": 1 + len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler))]),
            "lineno": node.lineno
        }
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)

    def visit_Assign(self, node):
        if self.current_class is None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id.isupper():
                        self.configs.append(target.id)
                    else:
                        self.globals.append(target.id)
        self.generic_visit(node)

def scan_files(root: Path):
    file_stats = {}
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        
        for file in filenames:
            if file.endswith(".py"):
                path = Path(dirpath) / file
                rel_path = str(path.relative_to(root))
                
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    analyzer = CodeAnalyzer()
                    analyzer.visit(tree)
                    
                    file_stats[rel_path] = {
                        "classes": analyzer.classes,
                        "functions": analyzer.functions,
                        "imports": sorted(list(set(analyzer.imports))),
                        "globals": analyzer.globals,
                        "configs": analyzer.configs,
                        "size": len(content.splitlines())
                    }
                except Exception as e:
                    print(f"Error parsing {rel_path}: {e}")
                    
    return file_stats

def generate_insights(stats: Dict[str, Any]):
    insights = {
        "complexity_hotspots": [],
        "dependency_graph": {},
        "architectural_notes": []
    }
    
    for f, data in stats.items():
        for func, info in data["functions"].items():
            if info["complexity"] > 10:
                insights["complexity_hotspots"].append({
                    "file": f,
                    "function": func,
                    "complexity": info["complexity"]
                })
        
        insights["dependency_graph"][f] = data["imports"]
        
    insights["complexity_hotspots"].sort(key=lambda x: x["complexity"], reverse=True)
    
    with open(OUTPUT_DIR / "insights.json", "w") as f:
        json.dump(insights, f, indent=2)
    print(f"✓ Generated {OUTPUT_DIR}/insights.json")
    return insights

def generate_variable_map(stats: Dict[str, Any]):
    vmap = {
        "global_state": [],
        "configuration": [],
        "interactions": []
    }
    
    for f, data in stats.items():
        for g in data["globals"]:
            vmap["global_state"].append({"file": f, "variable": g})
        for c in data["configs"]:
            vmap["configuration"].append({"file": f, "parameter": c})
            
    with open(OUTPUT_DIR / "variable_map.json", "w") as f:
        json.dump(vmap, f, indent=2)
    print(f"✓ Generated {OUTPUT_DIR}/variable_map.json")
    return vmap

def generate_feedback_log(insights, vmap):
    feedback = {
        "timestamp": datetime.now().isoformat(),
        "critical_issues": [],
        "recommendations": []
    }
    
    hotspots = insights.get("complexity_hotspots", [])
    if len(hotspots) > 5:
        feedback["critical_issues"].append(f"High cyclomatic complexity in {len(hotspots)} functions.")
        feedback["recommendations"].append("Refactor top 5 complex functions to reduce complexity below 10.")
    
    for hs in hotspots[:3]:
        feedback["recommendations"].append(f"Consider refactoring {hs['function']} (complexity: {hs['complexity']})")
        
    with open(OUTPUT_DIR / "feedback_log.json", "w") as f:
        json.dump(feedback, f, indent=2)
    print(f"✓ Generated {OUTPUT_DIR}/feedback_log.json")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Adaptive Code Analysis Workflow ===\n")
    
    print("Phase 0: Insight Mining & Deconstruction")
    print("  Step 2: Scanning Codebase...")
    stats = scan_files(PROJECT_ROOT)
    print(f"  Found {len(stats)} Python files")
    
    print("  Step 3: Extracting Insights...")
    insights = generate_insights(stats)
    
    print("\nPhase 1: Variable Universe Integration")
    print("  Step 4-6: Mapping Variables...")
    vmap = generate_variable_map(stats)
    
    print("\nPhase 4: Validation & Redesign")
    print("  Step 15: Generating Feedback Log...")
    generate_feedback_log(insights, vmap)
    
    print("\n=== Analysis Complete ===")
