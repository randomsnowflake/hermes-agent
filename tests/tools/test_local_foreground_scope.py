"""Foreground gateway commands must not consume the messaging cgroup."""
import os
from unittest.mock import patch

import pytest

from tools.environments.local import LocalEnvironment
from tools import process_registry
from tools.environments import local as local_env


@pytest.fixture(autouse=True)
def no_real_hermes_bin_dir(monkeypatch):
    # Resolving the console-script dir reads the real install; irrelevant to scoping.
    monkeypatch.setattr(local_env, "_resolve_hermes_bin_dir", lambda: None)


def bare_environment(tmp_path):
    env = object.__new__(LocalEnvironment)
    env.cwd = str(tmp_path)
    env.env = {}
    return env


@pytest.mark.platforms("linux")
def test_foreground_gateway_command_uses_worker_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "test-managed-gateway")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    with patch("tools.environments.local.subprocess.Popen") as popen:
        proc = popen.return_value
        proc.pid = os.getpid()
        bare_environment(tmp_path)._run_bash("printf scoped")
    args = popen.call_args.args[0]
    assert "--scope" in args
    assert any(arg.startswith("MemoryMax=") for arg in args)
    assert args[-2:] == ["-c", "printf scoped"]
    assert popen.call_args.kwargs["start_new_session"] is True


@pytest.mark.platforms("linux")
def test_foreground_gateway_fails_closed_without_user_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "test-managed-gateway")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)
    with patch("tools.environments.local.subprocess.Popen") as popen:
        with pytest.raises(process_registry.RestartSafeScopeUnavailable):
            bare_environment(tmp_path)._run_bash("printf unsafe")
        popen.assert_not_called()


def test_unmanaged_foreground_keeps_direct_shell(tmp_path, monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    with patch("tools.environments.local.subprocess.Popen") as popen:
        popen.return_value.pid = os.getpid()
        bare_environment(tmp_path)._run_bash("printf direct")
    assert "--scope" not in popen.call_args.args[0]
    assert popen.call_args.args[0][-2:] == ["-c", "printf direct"]


@pytest.mark.platforms("linux")
def test_live_worker_oom_does_not_kill_caller(tmp_path, monkeypatch):
    import json
    import shlex
    import sys
    from pathlib import Path

    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("requires a reachable systemd user bus")
    monkeypatch.setenv("INVOCATION_ID", "test-managed-gateway")
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "64")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    scope_argv = process_registry._systemd_scope_argv

    def no_swap_scope(*args):
        argv = scope_argv(*args)
        separator = argv.index('--')
        return argv[:separator] + ['--property', 'MemorySwapMax=0'] + argv[separator:]

    monkeypatch.setattr(process_registry, '_systemd_scope_argv', no_swap_scope)
    parent_cgroup = Path('/proc/self/cgroup').read_text()
    script = (
        "import json; from pathlib import Path; "
        "group=Path('/proc/self/cgroup').read_text().strip(); "
        "maximum=(Path('/sys/fs/cgroup') / group.split('::',1)[1].lstrip('/') / 'memory.max').read_text().strip(); "
        "print(json.dumps({'cgroup':group,'memory_max':maximum}),flush=True); "
        "allocation=bytearray(256*1024*1024)"
    )
    env = bare_environment(tmp_path)
    proc = env._run_bash(f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}")
    output, _ = proc.communicate(timeout=30)
    record = json.loads(output.splitlines()[0])
    assert 'hermes-worker-terminal-' in record['cgroup']
    assert int(record['memory_max']) == 64 * 1024 * 1024
    assert proc.returncode != 0
    assert Path('/proc/self/cgroup').read_text() == parent_cgroup
    survivor = env._run_bash('printf survived')
    output, _ = survivor.communicate(timeout=15)
    assert survivor.returncode == 0
    assert output == 'survived'
