"""Real registry → session kernel → socket RPC → read_file budget contracts."""

import json

import pytest

from tools import code_execution_tool
from tools.code_kernel import shutdown_all_kernels
from tools.registry import registry


@pytest.fixture(autouse=True)
def local_kernel(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def run_reads(path, budget, monkeypatch):
    monkeypatch.setattr(code_execution_tool, "_load_config", lambda: {
        "mode": "strict", "timeout": 30, "max_tool_calls": budget,
    })
    code = (
        "import json\n"
        "from hermes_tools import read_file\n"
        "cell_number = globals().get('cell_number', 0) + 1\n"
        # Distinct regions avoid read_file's repeated-read deduplication guard.
        f"print(json.dumps([read_file({str(path)!r}, offset=cell_number * 3 + i, limit=1) "
        "for i in range(3)]))\n"
        "print(cell_number)\n"
    )
    raw = registry.dispatch(
        "execute_code", {"code": code}, task_id="budget-test",
        enabled_tools=["read_file"],
    )
    assert isinstance(raw, str), raw
    return json.loads(raw)


def assert_reads(result, budget, cell_number):
    # A returned tool error is data, not a Python exception: outer success is expected.
    assert result["status"] == "success", result
    assert result["tool_calls_made"] == budget, result
    lines = result["output"].splitlines()
    replies = json.loads(lines[0])
    assert lines[1] == str(cell_number)
    for reply in replies[:budget]:
        assert "error" not in reply, reply
        assert "budget fixture" in reply.get("content", ""), reply
    for reply in replies[budget:]:
        assert f"Tool call limit reached ({budget})" in reply["error"], reply
    assert result["kernel"]["reused"] == (cell_number > 1)


@pytest.mark.parametrize("budget", [0, 2])
def test_each_execution_resets_usage_but_preserves_zero_limit(tmp_path, monkeypatch, budget):
    path = tmp_path / "input.txt"
    path.write_text("budget fixture\n" * 30, encoding="utf-8")
    for cell_number in range(1, 4):
        assert_reads(run_reads(path, budget, monkeypatch), budget, cell_number)


def test_reused_kernel_uses_current_cell_budget_without_losing_state(tmp_path, monkeypatch):
    path = tmp_path / "input.txt"
    path.write_text("budget fixture\n" * 30, encoding="utf-8")
    for cell_number, budget in enumerate([0, 2, 1, 0, 2], start=1):
        assert_reads(run_reads(path, budget, monkeypatch), budget, cell_number)
