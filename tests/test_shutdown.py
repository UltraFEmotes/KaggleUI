"""Exercise cancellation and notebook shutdown with real OS processes."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import httpx
import psutil
import pytest
from websockets.sync.client import connect

from kaggleui import launcher
from kaggleui.engines.context import EngineContext
from kaggleui.engines.download import hf_download
from kaggleui.proc import ManagedProcess, run_cmd
from kaggleui.util import free_port


async def wait_file(path):
    async def poll():
        while not path.exists():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(poll(), timeout=5)
    return int(path.read_text())


@pytest.mark.parametrize("shell", [False, True])
async def test_cancel_command_stops_process(tmp_path, shell):
    marker = tmp_path / "pid"
    code = "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(os.environ['KUI_TEST_PID']).write_text(str(os.getpid())); time.sleep(60)"
    cmd = [sys.executable, "-c", code]
    if shell:
        import shlex
        cmd = shlex.join(cmd)
    task = asyncio.create_task(run_cmd(cmd, env={"KUI_TEST_PID": str(marker)}))
    pid = await wait_file(marker)
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert not launcher._alive(pid)
    finally:
        if launcher._alive(pid):
            os.kill(pid, signal.SIGKILL)


async def test_stop_cleans_children_after_group_leader_exits(tmp_path):
    marker = tmp_path / "child"
    child = "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(os.environ['KUI_TEST_PID']).write_text(str(os.getpid())); time.sleep(60)"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(60)"
    proc = ManagedProcess("test", [sys.executable, "-c", parent, child], env={"KUI_TEST_PID": str(marker)})
    await proc.start()
    pid = await wait_file(marker)
    try:
        os.kill(proc.proc.pid, signal.SIGTERM)
        await asyncio.wait_for(proc.wait(), timeout=3)
        assert launcher._alive(pid)
        await proc.stop(timeout=0.2)
        for _ in range(50):
            if not launcher._alive(pid):
                break
            await asyncio.sleep(0.02)
        assert not launcher._alive(pid)
    finally:
        if launcher._alive(pid):
            os.kill(pid, signal.SIGKILL)


async def test_hf_download_cancellation_stops_worker(paths, hw, tmp_path, monkeypatch):
    # An offline stand-in blocks just like a large model download.
    fake = tmp_path / "huggingface_hub.py"
    fake.write_text("import os,time,pathlib\ndef snapshot_download(**kwargs):\n pathlib.Path(os.environ['KUI_TEST_PID']).write_text(str(os.getpid()))\n time.sleep(60)\n return '/snapshot'\n")
    marker = tmp_path / "download-pid"
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("KUI_TEST_PID", str(marker))
    ctx = EngineContext(paths, "test", "test", [], hw(), lambda _: None, lambda *_: None, "test", 8999)
    task = asyncio.create_task(hf_download(ctx, "test/model"))
    pid = await wait_file(marker)
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert not launcher._alive(pid)
        assert ctx.processes and not ctx.processes[0].running
    finally:
        if launcher._alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_notebook_stop_with_open_websocket_and_relaunch(paths):
    port = free_port()
    home = str(paths.home)
    state = launcher.launch(block=False, tunnel="none", mock=True, fake_hw="t4x2", port=port, home=home)
    try:
        headers = {"Authorization": f"Bearer {state['admin_token']}"}
        httpx.post(f"http://127.0.0.1:{port}/_kui/api/deployments", headers=headers, json={
            "modality": "text", "model": {"source": "ollama", "id": "qwen3", "variant": "0.6b", "engine": "ollama"},
        }).raise_for_status()
        reconnected = launcher.restart_tunnel(home)
        assert reconnected["pid"] == state["pid"]
        assert len(httpx.get(f"http://127.0.0.1:{port}/_kui/api/deployments", headers=headers).json()) == 1
        with connect(f"ws://127.0.0.1:{port}/_kui/ws", additional_headers=headers) as ws:
            assert json.loads(ws.recv(timeout=5))["type"] == "overview"
            start = time.monotonic()
            assert launcher.stop(home)
            assert time.monotonic() - start < 10
        assert not launcher._alive(state["pid"])
        assert json.loads(paths.state_file.read_text())["status"] == "stopped"
        assert launcher.stop(home) is False
        restarted = launcher.launch(block=False, tunnel="none", mock=True, fake_hw="t4x2", port=port, home=home)
        assert restarted["pid"] != state["pid"]
        assert httpx.get(f"http://127.0.0.1:{port}/_kui/api/session").status_code == 200
    finally:
        launcher.stop(home)


def test_forced_stop_cleans_detached_children(paths, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "STOP_TIMEOUT", 0.2)
    marker = tmp_path / "detached-pid"
    child = "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(os.environ['KUI_TEST_PID']).write_text(str(os.getpid())); time.sleep(60)"
    parent = "import subprocess,sys,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); subprocess.Popen([sys.executable,'-c',sys.argv[1]],start_new_session=True); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", parent, child], start_new_session=True,
                            env={**os.environ, "KUI_TEST_PID": str(marker)})
    pid = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pid = int(marker.read_text())
        paths.state_file.write_text(json.dumps({"pid": proc.pid, "status": "running",
                                               "process_started_at": psutil.Process(proc.pid).create_time()}))
        assert launcher.stop(str(paths.home))
        assert not launcher._alive(proc.pid)
        assert not launcher._alive(pid)
    finally:
        if pid and launcher._alive(pid):
            os.kill(pid, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_stale_state_does_not_stop_unrelated_process(paths):
    paths.state_file.write_text(json.dumps({"pid": os.getpid(), "status": "running",
                                           "process_started_at": psutil.Process().create_time() - 10}))
    assert launcher.stop(str(paths.home)) is False


async def test_retained_tunnel_is_adopted_and_full_stop_closes_it(paths, monkeypatch):
    from unittest.mock import AsyncMock

    from kaggleui.server.auth import Auth
    from kaggleui.tunnels import manager
    from kaggleui.tunnels.persistent import read_record

    # A real child writes to the same kind of regular log file as cloudflared.
    executable = paths.bin / "fake-cloudflared"
    executable.write_text(f"#!{sys.executable}\nimport time\n"
                          "print('https://retained.trycloudflare.com', flush=True)\n"
                          "print('Registered tunnel connection', flush=True)\n"
                          "while True:\n time.sleep(.1)\n print('still connected', flush=True)\n")
    executable.chmod(0o700)
    monkeypatch.setattr(manager, "ensure_cloudflared", AsyncMock(return_value=str(executable)))
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", AsyncMock(return_value=[]))
    first = manager.TunnelManager(paths)
    second = manager.TunnelManager(paths)
    try:
        await first.open("main", 8999, "cloudflare")
        url = await first.wait_url("main", timeout=5)
        assert url == "https://retained.trycloudflare.com"
        pid = read_record(paths)["pid"]
        original_auth = Auth(paths.home / "session.key")
        first.get("main").proc.save({"credentials": {"admin_token": original_auth.admin_token,
                                                    "password": original_auth.password, "api_key": original_auth.api_key}})
        await first.close_all(preserve_main=True)
        before = (paths.logs / "tunnel-main.log").stat().st_size
        await asyncio.sleep(.3)
        assert launcher._alive(pid)
        assert (paths.logs / "tunnel-main.log").stat().st_size > before
        await second.open("main", 8999, "cloudflare")
        assert await second.wait_url("main", timeout=5) == url
        assert read_record(paths)["pid"] == pid
        restored_auth = Auth(paths.home / "session.key")
        assert restored_auth.admin_token == original_auth.admin_token
        assert restored_auth.admin_cookie_value() == original_auth.admin_cookie_value()
        assert restored_auth.password == original_auth.password
        assert restored_auth.api_key == original_auth.api_key
        # Even without a live app server, the public stop helper closes the tunnel.
        assert await asyncio.to_thread(launcher.stop, str(paths.home))
        assert not launcher._alive(pid)
        assert not read_record(paths)
    finally:
        await first.close_all()
        await second.close_all()


def test_update_stop_retains_tunnel_child_then_full_stop_removes_it(paths, monkeypatch):
    from kaggleui.tunnels.persistent import read_record

    monkeypatch.setenv("NGROK_AUTHTOKEN", "test-token")
    executable = paths.bin / "ngrok"
    executable.write_text(f"#!{sys.executable}\nimport time\n"
                          "print('{\"url\": \"https://retained.example\"}', flush=True)\n"
                          "while True:\n time.sleep(.1)\n print('{}', flush=True)\n")
    executable.chmod(0o700)
    port = free_port()
    home = str(paths.home)
    pid = None
    try:
        first = launcher.launch(block=False, tunnel="ngrok", mock=True, fake_hw="t4x2", port=port, home=home)
        pid = read_record(paths)["pid"]
        assert launcher.stop(home, _keep_tunnel=True)
        assert not launcher._alive(first["pid"])
        assert launcher._alive(pid)
        second = launcher.launch(block=False, tunnel="ngrok", mock=True, fake_hw="t4x2", port=port, home=home)
        assert second["pid"] != first["pid"]
        assert read_record(paths)["pid"] == pid
        assert second["url"] == first["url"] == "https://retained.example"
        assert second["admin_token"] == first["admin_token"]
        assert second["password"] == first["password"]
        assert second["api_key"] == first["api_key"]
        assert launcher.stop(home)
        assert not launcher._alive(pid)
        assert not launcher.stop(home)
    finally:
        launcher.stop(home)
        if pid and launcher._alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_stale_tunnel_record_does_not_stop_unrelated_process(paths):
    record = {"pid": os.getpid(), "process_started_at": psutil.Process().create_time() - 10}
    (paths.home / "tunnel-main.json").write_text(json.dumps(record))
    assert launcher.stop(str(paths.home)) is False
    assert not (paths.home / "tunnel-main.json").exists()
