"""The browse corruption retry must preserve results and bounded failure semantics."""

from contextlib import contextmanager
import sqlite3

import pytest

from hermes_state import SessionDB


class FailingRead:
    def __init__(self, conn, error, phase="execute", repeat=False):
        self.conn, self.error, self.phase, self.repeat = conn, error, phase, repeat
        self.calls = 0
        self.handlers = []

    def execute(self, sql, params):
        self.calls += 1
        if self.error is not None and (self.calls == 1 or self.repeat):
            if self.phase == "execute":
                raise self.error
            return self
        return self.conn.execute(sql, params)

    def fetchall(self):
        raise self.error

    def set_progress_handler(self, callback, steps):
        self.handlers.append(callback)
        self.conn.set_progress_handler(callback, steps)


@contextmanager
def _read(reader):
    yield reader


@pytest.mark.parametrize("phase", ["execute", "fetchall"])
@pytest.mark.parametrize("code", [sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_CORRUPT | (3 << 8)])
def test_corrupt_preview_retry_preserves_browse_results(tmp_path, monkeypatch, phase, code):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("root", source="cli")
        db.append_message("root", role="user", content="old root")
        db.end_session("root", "compression")
        db.create_session("tip", source="cli", parent_session_id="root")
        db.append_message("tip", role="assistant", content="not a user preview")
        db.append_message("tip", role="user", content="earliest real user")
        db.append_message("tip", role="user", content="later user")
        db.create_session("peer", source="cli")
        db.append_message("peer", role="user", content="peer preview")
        db.create_session("excluded", source="cron")
        db.append_message("excluded", role="user", content="excluded")
        assert db._conn is not None
        reader = FailingRead(db._conn, None)
        monkeypatch.setattr(db, "_read_ctx", lambda: _read(reader))
        expected = db.list_recent_sessions_bounded(limit=5, exclude_sources=["cron"])
        assert reader.calls == 1  # Healthy databases keep the single-query fast path.
        assert {r["id"] for r in expected} == {"tip", "peer"}
        assert next(r for r in expected if r["id"] == "tip")["preview"] == "earliest real user"
        before = db._conn.total_changes
        error = sqlite3.DatabaseError("damaged secondary index")
        error.sqlite_errorcode = code
        reader = FailingRead(db._conn, error, phase)
        monkeypatch.setattr(db, "_read_ctx", lambda: _read(reader))
        assert db.list_recent_sessions_bounded(limit=5, exclude_sources=["cron"]) == expected
        assert reader.calls == 2
        assert reader.handlers[-1] is None
        assert db._conn.total_changes == before


@pytest.mark.parametrize("code,repeat,timeout", [
    (sqlite3.SQLITE_BUSY, False, 3),
    (sqlite3.SQLITE_IOERR, False, 3),
    (None, False, 3),
    (sqlite3.SQLITE_CORRUPT, True, 3),
    (sqlite3.SQLITE_CORRUPT, False, 0),
])
def test_browse_retry_keeps_errors_and_deadline(tmp_path, monkeypatch, code, repeat, timeout):
    with SessionDB(tmp_path / "state.db") as db:
        for i in range(20):
            db.create_session(f"session-{i}", source="cli")
            db.append_message(f"session-{i}", role="user", content="preview")
        assert db._conn is not None
        error = sqlite3.DatabaseError("database disk image is malformed")
        if code is not None:
            error.sqlite_errorcode = code
        reader = FailingRead(db._conn, error, repeat=repeat)
        monkeypatch.setattr(db, "_read_ctx", lambda: _read(reader))
        with pytest.raises(TimeoutError if timeout == 0 else sqlite3.DatabaseError) as caught:
            db.list_recent_sessions_bounded(timeout_seconds=timeout)
        if timeout != 0:
            assert caught.value is error
        assert reader.calls == (2 if code == sqlite3.SQLITE_CORRUPT else 1)
        assert reader.handlers[-1] is None
        assert db._conn.execute("SELECT 1").fetchone()[0] == 1
