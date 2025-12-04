import os
from dataclasses import dataclass, field
from typing import Optional

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
class EmbeddingConfig:
    # Model names
    MINILM_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLIP_MODEL: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    GLINER_MODEL: str = "urchade/gliner_medium-v2.1"
    
    # Device
    DEVICE: str = "cuda" if os.getenv("HOLOGRAM_FORCE_CPU", "0") != "1" else "cpu"

class Config:
    core = CoreConfig()
    storage = StorageConfig()
    server = ServerConfig()
    gravity = GravityConfig()
    embedding = EmbeddingConfig()

# Backwards compatibility for existing imports
VECTOR_DIM = Config.core.VECTOR_DIM
SEED = Config.core.SEED
