"""Auth rules and gateway behaviour against the real app with mock engines."""

import time

import pytest
from fastapi.testclient import TestClient

from kaggleui.schemas import Settings
from kaggleui.server.app import create_app
from kaggleui.server.state import AppState


@pytest.fixture
def state(paths):
    s = Settings()
    s.exposure.provider = "none"
    return AppState(paths=paths, settings=s, port=8999, mock=True, fake_hw="t4x2")


@pytest.fixture
def client(state):
    with TestClient(create_app(state)) as c:
        yield c


def deploy(client, state, modality="text", model_id="qwen3", variant="0.6b", engine="ollama"):
    admin = {"Authorization": f"Bearer {state.auth.admin_token}"}
    spec = {"modality": modality, "model": {"source": "ollama" if engine == "ollama" else "hf", "id": model_id,
                                            "variant": variant, "engine": engine}}
    dep = client.post("/_kui/api/deployments", json=spec, headers=admin).json()
    for _ in range(100):
        cur = next(d for d in client.get("/_kui/api/deployments", headers=admin).json() if d["id"] == dep["id"])
        if cur["status"] == "ready":
            return cur
        time.sleep(0.1)
    raise AssertionError(cur)


def test_panel_requires_admin(client, state):
    assert client.get("/_kui/api/overview").status_code == 401
    assert client.post("/_kui/api/login", json={"token": "nope"}).status_code == 401
    r = client.post("/_kui/api/login", json={"token": state.auth.admin_token})
    assert r.status_code == 200
    assert client.get("/_kui/api/overview").status_code == 200  # cookie set


def test_token_link_sets_cookie_and_strips_token(client, state):
    r = client.get(f"/?token={state.auth.admin_token}", follow_redirects=False)
    assert r.status_code == 303 and "token" not in r.headers["location"]
    assert client.get("/_kui/api/session").json()["admin"] is True


def test_login_rate_limited(client):
    for _ in range(10):
        client.post("/_kui/api/login", json={"token": "bad"})
    assert client.post("/_kui/api/login", json={"token": "bad"}).status_code == 429


def test_api_key_required_and_toggles(client, state):
    deploy(client, state)
    body = {"model": "qwen3:0.6b", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 401
    key = {"Authorization": f"Bearer {state.auth.api_key}"}
    r = client.post("/v1/chat/completions", json=body, headers=key)
    assert r.status_code == 200 and r.json()["model"] == "qwen3:0.6b"
    # x-api-key works for the Anthropic surface
    r = client.post("/v1/messages", json={**body, "max_tokens": 10},
                    headers={"x-api-key": state.auth.api_key, "anthropic-version": "2023-06-01"})
    assert r.status_code == 200 and r.json()["type"] == "message"
    # Ollama surface is off by default
    assert client.post("/api/chat", json=body, headers=key).status_code == 403
    state.settings.api.enabled = False
    assert client.post("/v1/chat/completions", json=body, headers=key).status_code == 403


def test_unknown_model_falls_back_when_single_model(client, state):
    deploy(client, state)
    key = {"Authorization": f"Bearer {state.auth.api_key}"}
    r = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
                    headers=key)
    assert r.status_code == 200


def test_wrong_modality_gives_helpful_404(client, state):
    deploy(client, state)
    key = {"Authorization": f"Bearer {state.auth.api_key}"}
    r = client.post("/v1/images/generations", json={"model": "sdxl", "prompt": "x"}, headers=key)
    assert r.status_code == 404 and "No an image model" not in r.text and "image model" in r.json()["error"]["message"]


def test_studio_access_password(client, state):
    deploy(client, state)
    assert client.get("/_kui/api/studio/models").status_code == 401
    assert client.post("/_kui/api/access-login", json={"password": "wrong"}).status_code == 401
    assert client.post("/_kui/api/access-login", json={"password": state.auth.password}).status_code == 200
    assert client.get("/_kui/api/studio/models").status_code == 200
    # studio header + access cookie works even with the public API off
    state.settings.api.enabled = False
    r = client.post("/v1/chat/completions", headers={"x-kui-studio": "1"},
                    json={"model": "qwen3:0.6b", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200


def test_studio_open_when_password_disabled(client, state):
    state.settings.access.password_enabled = False
    assert client.get("/_kui/api/studio/models").status_code == 200


def test_studio_ws_streams_tokens(client, state):
    deploy(client, state)
    client.post("/_kui/api/access-login", json={"password": state.auth.password})
    with client.websocket_connect("/_kui/studio/ws") as ws:
        ws.send_json({"op": "chat", "id": "r1", "model": "qwen3:0.6b",
                      "messages": [{"role": "user", "content": "hello"}]})
        text, done = "", None
        while done is None:
            msg = ws.receive_json()
            if msg["op"] == "delta":
                text += msg.get("content", "")
            elif msg["op"] in ("done", "error"):
                done = msg
    assert done["op"] == "done" and "hello" in text


def test_preset_export(client, state):
    deploy(client, state)
    r = client.get("/_kui/api/preset", headers={"Authorization": f"Bearer {state.auth.admin_token}"}).json()
    assert r["preset"]["deployments"][0]["model"]["id"] == "qwen3"
    assert "kaggleui.launch(preset=" in r["cell"]


def test_spa_fallback_does_not_swallow_api(client):
    assert client.get("/v1/does-not-exist").status_code == 404
    assert client.get("/models").status_code == 200


def test_world_prompt_paints_first_frame_with_image_model(client, state):
    key = {"Authorization": f"Bearer {state.auth.api_key}"}
    deploy(client, state, modality="world", model_id="Etched/oasis-500m", variant=None, engine="world_recipe")
    body = {"model": "oasis-500m", "prompt": "a snowy taiga", "frames": 8}
    # No image model running: a clear error instead of silently ignoring the text.
    r = client.post("/v1/videos", json=body, headers=key)
    assert r.status_code == 400 and "doesn't take text" in r.json()["error"]["message"]
    # Without a prompt the world model uses its own sample frame.
    assert client.post("/v1/videos", json={"model": "oasis-500m", "frames": 8}, headers=key).status_code == 200

    deploy(client, state, modality="image", model_id="stabilityai/sd-turbo", variant=None, engine="diffusers")
    r = client.post("/v1/videos", json=body, headers=key)
    assert r.status_code == 200 and r.json()["start_frame_model"] == "sd-turbo"
    r = client.post("/v1/start-frames", json=body, headers=key)
    import base64

    assert r.status_code == 200 and base64.b64decode(r.json()["b64_json"])[:4] == b"\x89PNG"


def test_diffusers_world_prompt_goes_directly_to_world_runtime(client, state):
    key = {"Authorization": f"Bearer {state.auth.api_key}"}
    deploy(client, state, modality="world", model_id="nvidia/Cosmos-Predict2.5-2B", variant=None,
           engine="diffusers_world")
    r = client.post("/v1/videos", json={"model": "cosmos-predict2.5-2b", "prompt": "a snowy city", "frames": 8},
                    headers=key)
    assert r.status_code == 200
    assert "start_frame_model" not in r.json()
