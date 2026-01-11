import os
import sys
import json
import argparse
import threading
import uvicorn
import requests
from pathlib import Path
from typing import List, Dict, Any

print("DEBUG: Script started")

# Ensure we can import the local package relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

class AgentMemory:
    def __init__(self, project_name: str = None, remote_url: str = None):
        # 1. Try to load from .hologram file in CWD
        config_path = Path(os.getcwd()) / ".hologram"
        if config_path.exists() and not project_name:
            try:
                data = json.loads(config_path.read_text())
                if "project" in data:
                    print(f"Found .hologram config: using project '{data['project']}'")
                    project_name = data["project"]
            except Exception as e:
                print(f"Warning: Failed to read .hologram: {e}")
        
        # 2. Fallback to default if still None
        if not project_name:
            project_name = "agent_self_memory"

        self.project_name = project_name
        self.remote_url = remote_url.rstrip('/') if remote_url else None
        
        if self.remote_url:
            print(f"Initialized AgentMemory in REMOTE mode (connecting to {self.remote_url})")
            # Verify connection
            try:
                r = requests.get(f"{self.remote_url}/")
                r.raise_for_status()
                print("Connected to Hologram Server successfully.")
            except Exception as e:
                print(f"Warning: Could not connect to remote server: {e}")
        else:
            # LOCAL MODE - Import Hologram here to avoid dependency if not needed
            from hologram import Hologram
            from hologram.config import Config
            
            print(f"Initialized AgentMemory in LOCAL mode")
            
            # Determine storage path
            self.memory_root = Path(Config.storage.MEMORY_DIR)
            self.project_dir = self.memory_root / self.project_name
            self.db_path = self.project_dir / Config.storage.SQLITE_DB_NAME
            
            # Ensure directory exists
            self.project_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize or Load
            if self.db_path.exists():
                print(f"Loading existing memory from: {self.db_path}")
                self.holo = Hologram.load(
                    str(self.db_path),
                    encoder_mode="minilm",
                    use_gravity=True,
                    auto_ingest_system=True
                )
            else:
                print(f"Initializing new Hologram memory for project: {project_name}")
                self.holo = Hologram.init(
                    encoder_mode="minilm", 
                    use_gravity=True, 
                    use_clip=False
                )
                # Force project name association
                self.holo.project = project_name

    def init_project(self):
        """Creates a .hologram marker file in the current directory."""
        config_path = Path(os.getcwd()) / ".hologram"
        data = {"project": self.project_name}
        config_path.write_text(json.dumps(data, indent=2))
        print(f"Initialized project '{self.project_name}' in {config_path}")

    def remember(self, text: str, source: str = "manual"):
        """
        Ingests a text fact or observation.
        """
        print(f"Remembering: {text[:50]}...")
        
        if self.remote_url:
            # Remote Ingest
            payload = {
                "project": self.project_name,
                "text": text,
                "origin": source,
                "tier": 1
            }
            res = requests.post(f"{self.remote_url}/ingest", json=payload)
            res.raise_for_status()
        else:
            # Local Ingest
            self.holo.add_text(source, text)
            
        print("Done.")

    def learn_codebase(self, root_path: str):
        """
        Walks a directory and ingests python files.
        """
        print(f"Learning codebase at: {root_path}")
        count = 0
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    # Skip venv and hidden folders
                    if ".venv" in full_path or "/." in full_path:
                        continue
                    
                    try:
                        # Ingest code
                        if self.remote_url:
                            # Read content and send
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            payload = {
                                "project": self.project_name,
                                "path": full_path, # Send path as metadata
                                "content": content
                            }
                            res = requests.post(f"{self.remote_url}/ingest/code", json=payload)
                            res.raise_for_status()
                            data = res.json()
                            num = data.get("concepts_extracted", 0)
                        else:
                            # Local Ingest
                            num = self.holo.ingest_code(full_path)
                            
                        count += num
                        print(f"  Ingested {file}: {num} concepts")
                    except Exception as e:
                        print(f"  Failed to ingest {file}: {e}")
        print(f"Total code concepts learned: {count}")

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves context based on a natural language query.
        """
        print(f"Recalling: {query}")
        
        output = []
        
        if self.remote_url:
            payload = {
                "project": self.project_name,
                "text": query,
                "top_k": top_k
            }
            res = requests.post(f"{self.remote_url}/query", json=payload)
            res.raise_for_status()
            data = res.json()
            # Map remote response structure to expected output
            for node in data.get("nodes", []):
                output.append({
                    "text": node.get("name"),
                    "score": node.get("score"),
                    "metadata": node
                })
        else:
            results = self.holo.search_text(query, top_k=top_k)
            for trace, score in results:
                text = getattr(trace, 'text', str(trace))
                metadata = getattr(trace, 'metadata', {})
                output.append({
                    "text": text,
                    "score": float(score),
                    "metadata": metadata
                })
            
        return output

    def query_code(self, query: str, top_k: int = 5):
        """
        Specifically queries the code knowledge base.
        """
        print(f"Querying code: {query}")
        
        if self.remote_url:
            payload = {
                "project": self.project_name,
                "text": query,
                "top_k": top_k
            }
            res = requests.post(f"{self.remote_url}/query/code", json=payload)
            res.raise_for_status()
            return res.json().get("results", [])
        else:
            results = self.holo.query_code(query, top_k=top_k)
            return results

    def save(self):
        """
        Persists the memory state.
        """
        if self.remote_url:
            print(f"Requesting remote save for {self.project_name}...")
            requests.post(f"{self.remote_url}/save/{self.project_name}")
        else:
            print(f"Saving memory to {self.db_path}...")
            self.holo.save(str(self.db_path))
        print("Saved.")

    def serve(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
        """
        Starts the Hologram API server.
        """
        if self.remote_url:
            print("Error: Cannot start server when configured as remote client.")
            return

        from hologram.server import start_server as launch_uvicorn_server
        
        print(f"Starting Hologram Server for project '{self.project_name}'...")
        # Ensure we save current state first so server picks it up
        self.save()
        
        launch_uvicorn_server(host=host, port=port, reload=reload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hologram Agent Skill Interface")
    parser.add_argument("--serve", action="store_true", help="Start the Hologram API Server")
    parser.add_argument("--test", action="store_true", help="Run the self-test suite")
    parser.add_argument("--init", action="store_true", help="Initialize this directory as a Hologram project")
    parser.add_argument("--project", type=str, default=None, help="Project name (overrides .hologram)")
    parser.add_argument("--ingest", type=str, help="Path to codebase to ingest")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for server (dev mode)")
    parser.add_argument("--remote", type=str, help="URL of remote Hologram server (e.g., http://localhost:8000)")
    
    args = parser.parse_args()
    
    agent = AgentMemory(project_name=args.project, remote_url=args.remote)
    
    if args.init:
        agent.init_project()
    elif args.ingest:
        agent.learn_codebase(args.ingest)
        agent.save()
    elif args.serve:
        agent.serve(reload=args.reload)
        
    elif args.test:
        if args.ingest:
             agent.learn_codebase(args.ingest)
        
        agent.remember("Hologram uses a gravity field where concepts are vectors with mass.")
        
        print("\n--- TEST: General Recall 'gravity' ---")
        hits = agent.recall("how does gravity work?")
        print(json.dumps(hits, indent=2))
        
        print("\n--- TEST: Code Query 'mitosis' ---")
        code_hits = agent.query_code("mitosis logic")
        print(json.dumps(code_hits, indent=2))
        
        agent.save()
    
    else:
        print("Please specify --init, --serve or --test")
        parser.print_help()
