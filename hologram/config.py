import os
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Dict, Any

@dataclass
class CoreConfig:
    VECTOR_DIM: int = 384  # Default for MiniLM
    SEED: int = 13
    DEBUG: bool = os.getenv("HOLOGRAM_DEBUG", "0") == "1"

@dataclass
class StorageConfig:
    # Directory for storing memory data
    MEMORY_DIR: str = os.getenv("HOLOGRAM_MEMORY_DIR", os.path.expanduser("~/.hologram_memory"))
    
    # FAISS settings
    USE_GPU: bool = os.getenv("HOLOGRAM_USE_GPU", "1") == "1"
    
    # SQLite settings
    USE_SQLITE: bool = True
    SQLITE_DB_NAME: str = "memory.db"
    
    # Global project name
    GLOBAL_PROJECT: str = "_hologram_system"

@dataclass
class ServerConfig:
    HOST: str = os.getenv("HOLOGRAM_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("HOLOGRAM_PORT", "8000"))
    RELOAD: bool = os.getenv("HOLOGRAM_RELOAD", "0") == "1"

@dataclass
class GravityConfig:
    # Physics parameters
    DEFAULT_MASS: float = 1.0
    DECAY_RATE: float = 0.001
    
    # Mitosis / Fusion
    MITOSIS_THRESHOLD: float = 0.3
    MITOSIS_MASS_THRESHOLD: float = 2.0
    FUSION_THRESHOLD: float = 0.85
    
    # Dynamic Gravity
    ENABLE_DYNAMICS: bool = True
    COOLDOWN_STEPS: int = 10

@dataclass
class CorefConfig:
    ENABLE_COREF: bool = True
    ENABLE_GRAVITY_FALLBACK: bool = True
    COREF_MODEL: str = "fastcoref"  # or specific model path

@dataclass
class EmbeddingConfig:
    # Model names
    MINILM_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLIP_MODEL: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    GLINER_MODEL: str = "urchade/gliner_medium-v2.1"
    
    # Device
    DEVICE: str = "cuda" if os.getenv("HOLOGRAM_FORCE_CPU", "0") != "1" else "cpu"



@dataclass
class EvolutionConfig:
    # Drift Thresholds
    DRIFT_SMALL: float = 0.08      # Fuse (Weighted Average)
    DRIFT_MEDIUM: float = 0.22     # Soft Fusion (Interpolation)
    DRIFT_LARGE: float = 0.38      # Mitosis (Split)
    
    # Decay & Protections
    OBSOLETE_DECAY: float = 0.60   # Forced decay factor for deprecated symbols
    DECAY_RATE: float = 0.1        # Mass decay rate per deprecation step
    Collision_THRESHOLD: float = 0.6  # Name similarity required for fusion
    ROT_THRESHOLD: float = 0.5     # Max allowed divergence from original vector before warning/action


class Config:
    """Centralized configuration with persistence support."""
    core = CoreConfig()
    storage = StorageConfig()
    server = ServerConfig()
    gravity = GravityConfig()
    coref = CorefConfig()
    embedding = EmbeddingConfig()
    evolution = EvolutionConfig() # NEW

    
    # Track if loaded from global config
    _loaded_from_global: bool = False
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Serialize all config sections to a flat dictionary."""
        result = {}
        for section_name in ["core", "storage", "server", "gravity", "coref", "embedding", "evolution"]:
            section = getattr(cls, section_name)
            for f in fields(section):
                key = f"{section_name}.{f.name}"
                result[key] = getattr(section, f.name)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], apply_env_overrides: bool = True):
        """
        Update config from a flat dictionary.
        If apply_env_overrides is True, environment variables take precedence.
        """
        sections = {
            "core": cls.core,
            "storage": cls.storage,
            "server": cls.server,
            "gravity": cls.gravity,
            "coref": cls.coref,
            "embedding": cls.embedding,
            "evolution": cls.evolution,
        }
        
        for key, value in data.items():
            if "." not in key:
                continue
            section_name, field_name = key.split(".", 1)
            if section_name not in sections:
                continue
            section = sections[section_name]
            if not hasattr(section, field_name):
                continue
            
            # Check if env var override exists
            env_key = f"HOLOGRAM_{field_name}"
            if apply_env_overrides and env_key in os.environ:
                continue  # Skip, env var takes precedence
            
            # Type conversion
            current_value = getattr(section, field_name)
            if isinstance(current_value, bool):
                value = str(value).lower() in ("1", "true", "yes")
            elif isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            
            setattr(section, field_name, value)
        
        cls._loaded_from_global = True
    
    @classmethod
    def diff(cls, other_dict: Dict[str, Any]) -> Dict[str, tuple]:
        """
        Compare current config with another dict.
        Returns dict of {key: (current_value, other_value)} for differences.
        """
        current = cls.to_dict()
        differences = {}
        all_keys = set(current.keys()) | set(other_dict.keys())
        for key in all_keys:
            curr_val = current.get(key)
            other_val = other_dict.get(key)
            if curr_val != other_val:
                differences[key] = (curr_val, other_val)
        return differences
    
    @classmethod
    def get_global_db_path(cls) -> str:
        """Get the path to the global config database."""
        return os.path.join(
            cls.storage.MEMORY_DIR,
            cls.storage.GLOBAL_PROJECT,
            cls.storage.SQLITE_DB_NAME
        )


# Backwards compatibility for existing imports
VECTOR_DIM = Config.core.VECTOR_DIM
SEED = Config.core.SEED

