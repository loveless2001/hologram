from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from hologram.api import Hologram
from hologram.chatbot import ChatMemory, ChatWindow, EchoProvider, SessionLog
from hologram.store import MemoryStore, Trace


def test_memory_store_persistence(tmp_path: Path):
    store = MemoryStore(vec_dim=4)
    trace = Trace(
        trace_id="t1",
        kind="text",
        content="hello world",
        vec=np.ones(4, dtype=np.float32),
        meta={"role": "user"},
    )
    store.add_trace(trace)
    store.link_trace("glyph:hello", trace.trace_id)

    path = tmp_path / "store.json"
    store.save(path.as_posix())

    loaded = MemoryStore.load(path.as_posix())
    assert loaded.vec_dim == store.vec_dim
    assert loaded.get_trace("t1").content == "hello world"
    assert loaded.get_glyph("glyph:hello").trace_ids == ["t1"]


def test_chat_memory_and_logs(tmp_path: Path, isolated_hologram):
    hologram = isolated_hologram
    memory = ChatMemory(hologram=hologram, session_window=5, cross_session_k=2)
    logs = SessionLog(tmp_path)

    provider = EchoProvider()
    window = ChatWindow(
        provider=provider,
        memory=memory,
        logs=logs,
        session_id="alpha",
        system_prompt="You are testing",
    )

    reply = window.step("Hello there")
    assert "Hello there" in reply

    log_entries = logs.read("alpha")
    assert len(log_entries) == 2  # user + assistant
    assert log_entries[0]["role"] == "user"
    assert log_entries[1]["role"] == "assistant"

    # Ensure cross-session memories are surfaced
    memory.record_message("beta", "user", "I also said hello")
    hits = memory.search_global_context("hello", exclude_session="alpha", top_k=1)
    assert hits and hits[0]["session_id"] == "beta"
