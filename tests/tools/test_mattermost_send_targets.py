"""Mattermost IDs must remain separate from explicit thread IDs in cron sends."""
from unittest.mock import patch

import pytest

from cron.scheduler_delivery import _resolve_delivery_targets
from tools.send_message_targets import _parse_target_ref, resolve_send_target
from tools.send_message_tool import prepare_send_message_platforms

# Synthetic 26-character lowercase alphanumeric IDs, not real conversations.
CHANNEL = "channel" + "a" * 19
THREAD = "thread" + "b" * 20


@pytest.fixture(autouse=True)
def load_platforms():
    prepare_send_message_platforms()


@pytest.mark.parametrize("thread", [None, THREAD])
def test_native_ids_are_explicit_without_directory_lookup(thread):
    ref = CHANNEL if thread is None else f"{CHANNEL}:{thread}"
    assert _parse_target_ref("mattermost", ref) == (CHANNEL, thread, True)
    with patch("gateway.channel_directory.resolve_channel_name") as directory:
        assert resolve_send_target("mattermost", ref) == (CHANNEL, thread, None)
    directory.assert_not_called()


@pytest.mark.parametrize("lane", ["deliver", "failure_deliver"])
def test_cron_explicit_thread_target_preserves_both_ids(lane):
    job = {"deliver": "local", lane: f"mattermost:{CHANNEL}:{THREAD}"}
    assert _resolve_delivery_targets(job, for_failure=lane == "failure_deliver") == [{
        "platform": "mattermost", "chat_id": CHANNEL, "thread_id": THREAD,
        "_resolved_from": "explicit",
    }]


def test_cron_channel_root_does_not_inherit_origin_thread():
    job = {"deliver": f"mattermost:{CHANNEL}", "origin": {
        "platform": "mattermost", "chat_id": CHANNEL, "thread_id": THREAD,
    }}
    assert _resolve_delivery_targets(job)[0]["thread_id"] is None


def test_channel_label_still_uses_directory():
    with patch("gateway.channel_directory.resolve_channel_name", return_value=CHANNEL) as directory:
        assert resolve_send_target("mattermost", "Home") == (CHANNEL, None, None)
    directory.assert_called_once_with("mattermost", "Home")


@pytest.mark.parametrize("ref", [
    f"{CHANNEL}:short", f"short:{THREAD}", f"{CHANNEL}:{THREAD}:extra",
    f"{CHANNEL.upper()}:{THREAD}", f"{CHANNEL}:",
])
def test_malformed_thread_ids_are_not_reinterpreted(ref):
    assert _parse_target_ref("mattermost", ref) == (None, None, False)
    with patch("gateway.channel_directory.resolve_channel_name", return_value=None):
        chat, thread, error = resolve_send_target("mattermost", ref)
    assert chat is None and thread is None and error
