"""
Global Configuration Management for Hologram.

Handles initialization, loading, and synchronization of the global
`_hologram_system` project which stores system-wide configuration.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from .config import Config


class SyncStatus(Enum):
    """Status of config synchronization."""
    SYNCED = "synced"           # Local and global match
    CONFLICT = "conflict"       # Differences exist
    GLOBAL_MISSING = "missing"  # No global config exists
    LOCAL_NEWER = "local_newer" # Local has changes not in global


def get_global_db_path() -> str:
    """Get the path to the global config database."""
    return Config.get_global_db_path()


def global_project_exists() -> bool:
    """Check if the global project database exists."""
    db_path = get_global_db_path()
    return os.path.exists(db_path)


def init_global_project(force: bool = False) -> Tuple[bool, str]:
    """
    Initialize the global project with current config.
    
    Args:
        force: If True, overwrite existing global config.
        
    Returns:
        (success, message)
    """
    db_path = get_global_db_path()
    db_dir = os.path.dirname(db_path)
    
    # Create directory if needed
    os.makedirs(db_dir, exist_ok=True)
    
    if os.path.exists(db_path) and not force:
        return False, f"Global project already exists at {db_path}"
    
    # Create/overwrite the database
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            # Create meta table for config
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Save system marker
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("_system_managed", "true")
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("_created_at", str(os.popen("date -Iseconds").read().strip()))
            )
            
            # Save all config values
            config_dict = Config.to_dict()
            for key, value in config_dict.items():
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    (f"config:{key}", json.dumps(value))
                )
        
        return True, f"Global project initialized at {db_path}"
    finally:
        conn.close()


def load_global_config() -> Optional[Dict[str, Any]]:
    """
    Load configuration from the global project database.
    
    Returns:
        Dict of config values, or None if global project doesn't exist.
    """
    db_path = get_global_db_path()
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT key, value FROM meta WHERE key LIKE 'config:%'"
        )
        
        config = {}
        for key, value in cursor:
            # Remove 'config:' prefix
            config_key = key[7:]  # len("config:") = 7
            config[config_key] = json.loads(value)
        
        return config if config else None
    finally:
        conn.close()


def save_global_config(config_dict: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Save configuration to the global project database.
    
    Args:
        config_dict: Config to save. If None, uses current Config.to_dict().
        
    Returns:
        (success, message)
    """
    if config_dict is None:
        config_dict = Config.to_dict()
    
    db_path = get_global_db_path()
    if not os.path.exists(db_path):
        return init_global_project()
    
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            for key, value in config_dict.items():
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    (f"config:{key}", json.dumps(value))
                )
            
            # Update timestamp
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("_updated_at", str(os.popen("date -Iseconds").read().strip()))
            )
        
        return True, "Global config updated"
    finally:
        conn.close()


def sync_config() -> Tuple[SyncStatus, Dict[str, tuple]]:
    """
    Synchronize local config with global config.
    
    Returns:
        (status, differences)
        - status: SyncStatus enum
        - differences: Dict of {key: (local_value, global_value)}
    """
    global_config = load_global_config()
    
    if global_config is None:
        return SyncStatus.GLOBAL_MISSING, {}
    
    differences = Config.diff(global_config)
    
    if not differences:
        return SyncStatus.SYNCED, {}
    
    return SyncStatus.CONFLICT, differences


def apply_global_config(apply_env_overrides: bool = True) -> Tuple[bool, str]:
    """
    Load and apply the global config to the current Config instance.
    Environment variables still take precedence if apply_env_overrides is True.
    
    Returns:
        (success, message)
    """
    global_config = load_global_config()
    
    if global_config is None:
        return False, "No global config found"
    
    Config.from_dict(global_config, apply_env_overrides=apply_env_overrides)
    return True, f"Applied {len(global_config)} config values from global project"


def is_system_managed() -> bool:
    """Check if the global project is marked as system-managed."""
    db_path = get_global_db_path()
    if not os.path.exists(db_path):
        return False
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT value FROM meta WHERE key = '_system_managed'"
        )
        row = cursor.fetchone()
        return row is not None and row[0] == "true"
    finally:
        conn.close()
