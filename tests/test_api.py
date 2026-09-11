"""HTTP + WebSocket protocol tests against the live app (services stubbed)."""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client():
    # The lifespan warmup is a no-op (see conftest.pytest_configure), so the
    # ASGI app boots instantly and offline.
    with TestClient(main.app) as c:
        yield c


class TestHttpApi:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["version"] == main.__version__
        assert body["active_ws_connections"] == 0
        assert "wake_strategy" in body
        assert "stt_model" in body
        assert "llm_model" in body
        assert "tts_voice" in body

    def test_dashboard_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Nova" in resp.text
        assert f"v{main.__version__}" in resp.text
        assert "/static/js/app.js" in resp.text

    def test_unknown_route_404(self, client):
        assert client.get("/nope").status_code == 404


class TestWebSocketProtocol:
    def test_connect_welcome_and_ping_pong(self, client):
        with client.websocket_connect("/ws") as ws:
            # First frame: connection system_event with wake strategy details.
            first = ws.receive_json()
            assert first["type"] == "system_event"
            assert first["status"] == "idle"

            # Ping -> pong echoes the client timestamp.
            ws.send_json({"type": "ping", "client_timestamp": 1234})
            pong = ws.receive_json()
            assert pong["type"] == "pong"
            assert pong["client_timestamp"] == 1234
            assert "server_timestamp" in pong

    def test_listening_state_roundtrip(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # consume welcome frame

            ws.send_json({"type": "start_listening"})
            state_change = ws.receive_json()
            assert state_change["type"] == "state_change"
            assert state_change["stage"] == "LISTENING"

            ws.send_json({"type": "stop_listening"})
            state_change = ws.receive_json()
            assert state_change["type"] == "state_change"
            assert state_change["stage"] == "IDLE"

    def test_empty_chat_message_is_warned(self, client):
        # The real chat pipeline runs as a background task over the socket;
        # that is covered deterministically in test_pipeline.py. Here we verify
        # the protocol path for a degenerate message that returns synchronously.
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome
            ws.send_json({"type": "chat_message", "content": "   "})
            event = ws.receive_json()
            assert event["type"] == "system_event"
            assert "Empty chat" in event.get("message", "")
            assert event["status"] == "warning"

    def test_interrupt_frame_acknowledged(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome
            ws.send_json({"type": "interrupt"})
            system_event = ws.receive_json()
            assert system_event["type"] == "system_event"
            assert "interrupt" in system_event.get("message", "").lower()

    def test_binary_garbage_does_not_crash(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # welcome
            # Odd-length frame is not Int16-aligned -> dropped by is_binary_frame.
            ws.send_bytes(b"\x00\x01\x02")
            ws.send_json({"type": "ping", "client_timestamp": 1})
            assert ws.receive_json()["type"] == "pong"
