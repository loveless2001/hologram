"""
Pytest configuration and shared fixtures for test isolation.
"""
import pytest
import shutil
from pathlib import Path

# Reset global state between tests
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test to ensure isolation."""
    # Clear hologram instances
    from hologram.server import hologram_instances
    hologram_instances.clear()
    
    # Reset Config to defaults (reload the module)
    # This ensures each test starts with clean config
    import importlib
    from hologram import config
    importlib.reload(config)
    
    yield
    
    # Cleanup after test
    hologram_instances.clear()


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Provide a temporary memory directory for tests."""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir(exist_ok=True)
    
    # Override Config memory dir
    from hologram.config import Config
    original_dir = Config.storage.MEMORY_DIR
    Config.storage.MEMORY_DIR = str(memory_dir)
    
    yield memory_dir
    
    # Restore original
    Config.storage.MEMORY_DIR = original_dir
    
    # Cleanup
    if memory_dir.exists():
        shutil.rmtree(memory_dir)


@pytest.fixture
def isolated_hologram():
    """Create an isolated Hologram instance for testing."""
    from hologram.api import Hologram
    
    # Use minilm instead of hash to avoid meta tensor issues
    holo = Hologram.init(use_clip=False, use_gravity=True, auto_ingest_system=False, encoder_mode="minilm")
    
    yield holo
    
    # Cleanup is handled by reset_global_state


@pytest.fixture
def test_client():
    """Provide a TestClient for API testing with clean state."""
    from fastapi.testclient import TestClient
    from hologram.server import app, hologram_instances
    
    # Clear instances before creating client
    hologram_instances.clear()
    
    client = TestClient(app)
    
    yield client
    
    # Cleanup
    hologram_instances.clear()
