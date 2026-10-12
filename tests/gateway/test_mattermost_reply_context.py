"""Replies to out-of-session Mattermost reports must carry the report text."""
import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.mattermost.adapter import MattermostAdapter

ROOT = "a" * 26
CHANNEL = "b" * 26


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_kind", ["own", "other", "wrong_channel", "missing", "deleted"])
async def test_reply_context_is_the_exact_same_channel_root(monkeypatch, parent_kind):
    monkeypatch.setenv("MATTERMOST_ALLOWED_CHANNELS", CHANNEL)
    monkeypatch.setenv("MATTERMOST_REQUIRE_MENTION", "false")
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="test", extra={"reply_mode": "thread"}))
    adapter._bot_user_id = "bot"
    parent = {"id": ROOT, "channel_id": CHANNEL, "message": "Full deployment report\n" + "details " * 1000,
              "user_id": "bot" if parent_kind == "own" else "monitor", "delete_at": 0}
    if parent_kind == "wrong_channel":
        parent["channel_id"] = "another-channel"
    elif parent_kind == "missing":
        parent = {}
    elif parent_kind == "deleted":
        parent["delete_at"] = 1
    adapter._api_get = AsyncMock(return_value=parent)
    adapter.handle_message = AsyncMock()
    post = {"id": "c" * 26, "user_id": "human", "channel_id": CHANNEL,
            "root_id": ROOT, "message": "What went wrong?"}
    await adapter._handle_ws_event({"event": "posted", "data": {
        "post": json.dumps(post), "channel_type": "P", "sender_name": "@human"}})
    adapter._api_get.assert_awaited_once_with("posts/" + ROOT)
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "What went wrong?"
    assert event.source.thread_id == ROOT
    if parent_kind in {"own", "other"}:
        assert event.reply_to_message_id == ROOT
        assert event.reply_to_text == parent["message"]
        assert event.reply_to_author_id == parent["user_id"]
        assert event.reply_to_is_own_message is (parent_kind == "own")
    else:
        assert event.reply_to_text is None
        assert event.reply_to_message_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("root,message", [("", "New conversation"), (ROOT, "/status"),
                                         ("../users/me", "Malformed root")])
async def test_root_posts_commands_and_invalid_roots_do_not_fetch_context(monkeypatch, root, message):
    monkeypatch.setenv("MATTERMOST_ALLOWED_CHANNELS", CHANNEL)
    monkeypatch.setenv("MATTERMOST_REQUIRE_MENTION", "false")
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="test", extra={"reply_mode": "thread"}))
    adapter._bot_user_id = "bot"
    adapter._api_get = AsyncMock()
    adapter.handle_message = AsyncMock()
    post = {"id": "c" * 26, "user_id": "human", "channel_id": CHANNEL,
            "root_id": root, "message": message}
    await adapter._handle_ws_event({"event": "posted", "data": {
        "post": json.dumps(post), "channel_type": "P", "sender_name": "@human"}})
    adapter._api_get.assert_not_awaited()
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, aiohttp.ClientConnectionError])
async def test_report_fetch_failure_does_not_drop_the_human_reply(monkeypatch, error):
    monkeypatch.setenv("MATTERMOST_ALLOWED_CHANNELS", CHANNEL)
    monkeypatch.setenv("MATTERMOST_REQUIRE_MENTION", "false")
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="test", extra={"reply_mode": "thread"}))
    adapter._bot_user_id = "bot"
    adapter._session = MagicMock()
    adapter._session.get.side_effect = error()
    adapter.handle_message = AsyncMock()
    post = {"id": "c" * 26, "user_id": "human", "channel_id": CHANNEL,
            "root_id": ROOT, "message": "What went wrong?"}
    await adapter._handle_ws_event({"event": "posted", "data": {
        "post": json.dumps(post), "channel_type": "P", "sender_name": "@human"}})
    assert adapter.handle_message.await_count == 1
    assert adapter.handle_message.await_args.args[0].reply_to_text is None
