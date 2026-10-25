"""Foreground terminal commands leave the gateway cgroup like background workers do."""

import pytest

import tools.process_registry as pr


@pytest.mark.platforms("linux")
def test_wraps_when_supervised_and_scope_available(monkeypatch):
    monkeypatch.setattr(pr, "_IS_LINUX", True)
    monkeypatch.setattr(pr, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(pr, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
    monkeypatch.setattr(pr, "systemd_user_bus_env", lambda env: {**env, "DBUS_SESSION_BUS_ADDRESS": "x"})

    argv, env = pr.foreground_scope_argv(["/bin/bash", "-c", "echo hi"], {"A": "1"})

    assert argv[:3] == ["/usr/bin/systemd-run", "--user", "--scope"]
    assert argv[argv.index("--unit") + 1].startswith("hermes-worker-fg-")
    assert argv[-4:] == ["--", "/bin/bash", "-c", "echo hi"]
    assert env == {"A": "1", "DBUS_SESSION_BUS_ADDRESS": "x"}


@pytest.mark.parametrize("supervised,available", [(False, True), (True, False)])
def test_passthrough_without_supervisor_or_scope(monkeypatch, supervised, available):
    monkeypatch.setattr(pr, "_IS_LINUX", True)
    monkeypatch.setattr(pr, "_is_supervised_gateway_process", lambda: supervised)
    monkeypatch.setattr(pr, "_systemd_run_user_scope_available", lambda: available)
    original = ["/bin/bash", "-c", "true"]

    argv, env = pr.foreground_scope_argv(original, {"A": "1"})

    assert argv == original and env == {"A": "1"}
