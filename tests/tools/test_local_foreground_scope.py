"""Local foreground spawning honors best-effort gateway cgroup isolation.

The live probe is opt-in (HERMES_RUN_E2E=1) and only reads
cgroup metadata. Never allocate to exhaustion: the direct fallback is intentional.
"""

import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from unittest.mock import patch

import pytest

from tools import process_registry
from tools.environments import local
from tools.environments.local import LocalEnvironment


def bare_environment(tmp_path):
    env = object.__new__(LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    return env


@pytest.fixture
def mock_spawn(monkeypatch):
    """Exercise argv wiring without discovering shells, credentials or a live bus."""
    monkeypatch.setattr(local, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(local, "_make_run_env", lambda env: {"TEST_CHILD": "1"})
    with patch("tools.environments.local.subprocess.Popen") as popen:
        popen.return_value.pid = os.getpid()
        yield popen


@pytest.mark.platforms("linux")
def test_foreground_gateway_command_uses_worker_scope(tmp_path, monkeypatch, mock_spawn):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/systemd-run")
    monkeypatch.setattr(process_registry, "_worker_memory_max_bytes", lambda: 128 * 1024 * 1024)
    monkeypatch.setattr(process_registry, "systemd_user_bus_env",
                        lambda env: {**env, "DBUS_SESSION_BUS_ADDRESS": "test-bus"})

    proc = bare_environment(tmp_path)._run_bash("printf scoped")

    mock_spawn.assert_called_once()
    assert proc is mock_spawn.return_value
    args = mock_spawn.call_args.args[0]
    assert args[:3] == ["/usr/bin/systemd-run", "--user", "--scope"]
    assert args[args.index("--unit") + 1].startswith("hermes-worker-fg-")
    assert "--collect" in args
    assert "MemoryAccounting=yes" in args
    assert f"MemoryMax={128 * 1024 * 1024}" in args
    assert args[args.index("--") + 1:] == ["/bin/bash", "-c", "printf scoped"]
    kwargs = mock_spawn.call_args.kwargs
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"] == {"TEST_CHILD": "1", "DBUS_SESSION_BUS_ADDRESS": "test-bus"}


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("scope_available", [False, True], ids=["unavailable", "binary-disappeared"])
def test_foreground_gateway_falls_back_to_direct_shell(tmp_path, monkeypatch, mock_spawn, scope_available):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: scope_available)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    bare_environment(tmp_path)._run_bash("printf direct")

    mock_spawn.assert_called_once()
    assert mock_spawn.call_args.args[0] == ["/bin/bash", "-c", "printf direct"]
    assert mock_spawn.call_args.kwargs["env"] == {"TEST_CHILD": "1"}
    assert mock_spawn.call_args.kwargs["start_new_session"] is True


@pytest.mark.platforms("posix")
def test_unmanaged_foreground_keeps_direct_shell(tmp_path, monkeypatch, mock_spawn):
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    with patch.object(process_registry, "_systemd_run_user_scope_available") as probe:
        bare_environment(tmp_path)._run_bash("printf direct")
    probe.assert_not_called()
    mock_spawn.assert_called_once()
    assert mock_spawn.call_args.args[0] == ["/bin/bash", "-c", "printf direct"]
    assert mock_spawn.call_args.kwargs["env"] == {"TEST_CHILD": "1"}


@pytest.mark.platforms("linux")
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("HERMES_RUN_E2E") != "1",
                    reason="opt in with HERMES_RUN_E2E=1 for the non-OOM scope probe")
def test_live_worker_scope_isolated_without_memory_pressure(tmp_path, monkeypatch):
    """Verify placement and a finite limit, not OOM behavior or OOM survival."""
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        pytest.skip("requires cgroup v2")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("requires bash")
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "128")
    # The real probe updates these cached compatibility flags; restore them on exit.
    for name in ("_SYSTEMD_SCOPE_AVAILABLE", "_SYSTEMD_SCOPE_PROBED_AT", "_SYSTEMD_RUN_NO_EXPAND"):
        monkeypatch.setattr(process_registry, name, getattr(process_registry, name))
    monkeypatch.setattr(process_registry, "_SYSTEMD_SCOPE_AVAILABLE", None)
    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("requires a reachable systemd user bus with scope support")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(local, "_find_bash", lambda: bash)
    monkeypatch.setattr(local, "_make_run_env", lambda env: process_registry.systemd_user_bus_env(
        {"PATH": os.defpath, "HOME": str(tmp_path)}))
    parent_cgroup = Path("/proc/self/cgroup").read_text()
    expected_limit = process_registry._worker_memory_max_bytes()
    script = (
        "import json, signal; signal.alarm(5); from pathlib import Path; "
        "group=Path('/proc/self/cgroup').read_text().strip(); "
        "relative=next(line[3:] for line in group.splitlines() if line.startswith('0::')); "
        "maximum=(Path('/sys/fs/cgroup') / relative.lstrip('/') / 'memory.max').read_text().strip(); "
        "print(json.dumps({'cgroup':group,'memory_max':maximum}),flush=True)"
    )
    env = bare_environment(tmp_path)
    # -I -S avoids loading site hooks; exec leaves no shell child to orphan.
    proc = env._run_bash(f"exec {shlex.quote(sys.executable)} -I -S -c {shlex.quote(script)}")
    try:
        output, _ = proc.communicate(timeout=15)
        assert proc.returncode == 0, output
        record = json.loads(output)
        assert "hermes-worker-fg-" in record["cgroup"]
        assert record["cgroup"] != parent_cgroup.strip()
        assert int(record["memory_max"]) == expected_limit
        assert Path("/proc/self/cgroup").read_text() == parent_cgroup
    finally:
        if proc.poll() is None:
            env._kill_process(proc)
        try:
            proc.wait(timeout=5)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
