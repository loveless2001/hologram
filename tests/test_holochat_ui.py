"""Smoke tests for the optional static holochat UI."""

import os

os.environ["HOLOGRAM_QUIET"] = "1"
os.environ["HOLOCHAT_DISABLE_LLM"] = "1"

from fastapi.testclient import TestClient

from apps.holochat.server import app


def test_ui_shell_serves_html():
    client = TestClient(app)

    response = client.get("/app")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Holochat" in response.text
    assert "/assets/app.js" in response.text


def test_ui_assets_are_served():
    client = TestClient(app)

    response = client.get("/assets/app.js")

    assert response.status_code == 200
    assert "sendMessage" in response.text
