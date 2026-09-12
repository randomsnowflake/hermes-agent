"""Manual tool runs retain scheduled identity and the real store's claim guards."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from cron import executions, jobs, scheduler
from cron.occurrences import completed_occurrence
from tools import cronjob_tools  # registers the production handler
from tools.registry import registry


@pytest.fixture
def manual_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clock = [datetime(2026, 9, 11, 12, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    effect = tmp_path / "effects.txt"
    script = tmp_path / "scripts" / "effect.py"
    script.parent.mkdir()
    script.write_text(
        "from pathlib import Path\n"
        f"with Path({str(effect)!r}).open('a') as f: f.write('effect\\n')\n"
        "print('done')\n"
    )
    job = jobs.create_job(
        prompt="", schedule="5 0 * * *", script=str(script),
        no_agent=True, deliver="local",
    )
    return job, effect, clock


def _manual_run(job):
    return json.loads(registry.dispatch(
        "cronjob_manage", {"action": "run", "job_id": job["id"]},
    ))


def test_repeated_manual_runs_leave_scheduled_occurrence_available(manual_job):
    job, effect, clock = manual_job
    slot = job["next_run_at"]
    for count in (1, 2):
        result = _manual_run(job)
        assert result["job"]["execution_success"] is True, result
        assert effect.read_text().splitlines() == ["effect"] * count
        assert jobs.get_job(job["id"])["next_run_at"] == slot
        rows = executions.list_executions(job_id=job["id"])
        assert len(rows) == count
        assert all(row["status"] == "completed" for row in rows)
        assert all(row["scheduled_instant"] is None for row in rows)
        assert not completed_occurrence(job, slot)

    # The real scheduled claim must still run once, then reject a rolled-back slot.
    clock[0] = datetime.fromisoformat(slot) + timedelta(seconds=1)
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert claimed["_scheduled_instant"] == slot
    assert scheduler.run_one_job(claimed) is True
    assert completed_occurrence(job, slot)
    assert effect.read_text().splitlines() == ["effect"] * 3
    records = jobs.load_jobs()
    records[0]["next_run_at"] = slot
    jobs.save_jobs(records)
    assert jobs.claim_job_for_fire(job["id"], return_job=True) is False
    assert effect.read_text().splitlines() == ["effect"] * 3


def test_manual_intent_preserves_paused_and_inflight_guards(manual_job):
    job, effect, _clock = manual_job
    jobs.pause_job(job["id"])
    paused = jobs.get_job(job["id"])
    result = _manual_run(job)
    assert result["job"]["executed"] is False, result
    assert jobs.get_job(job["id"]) == paused
    assert not effect.exists()

    jobs.resume_job(job["id"])
    claimed, error = cronjob_tools._claim_for_manual_run(job["id"], "test")
    assert error is None
    assert claimed["_scheduled_instant"] is None
    owner = claimed["fire_claim"]["by"]
    result = _manual_run(job)
    assert result["job"]["executed"] is False, result
    assert jobs.claim_job_for_fire(job["id"]) is False
    assert jobs.get_job(job["id"])["fire_claim"]["by"] == owner
    assert not effect.exists()
