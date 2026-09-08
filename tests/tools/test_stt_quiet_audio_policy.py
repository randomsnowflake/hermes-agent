"""Carried OpenAI policy: preserve quiet speech before trimming and retry sparsity."""

import math
import shutil
import struct
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from tools import transcription_audio as audio
from tools import transcription_tools as stt


@pytest.mark.parametrize("case", [
    "quiet", "loud", "silent", "disabled", "missing-ffmpeg", "encode-failure",
    "prepare-failure", "dispatch-failure", "other-provider",
])
def test_normalization_precedes_trim_and_cleans_up(tmp_path, monkeypatch, case):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real ffmpeg/ffprobe required")
    rate = 16000
    amplitude = 10000 if case == "loud" else (0 if case == "silent" else 100)
    source = tmp_path / "voice.wav"
    # Quiet speech-band energy around a pause, long enough to exercise the trim.
    samples = [int(amplitude * math.sin(2 * math.pi * 400 * i / rate))
               if i < 5 * rate or i >= 10 * rate else 0 for i in range(15 * rate)]
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    original = source.read_bytes()
    provider = "groq" if case == "other-provider" else "openai"
    config = {"stt": {"provider": provider, "openai": {
        "model": "gpt-transcribe", "normalize_quiet_audio": "false" if case == "disabled" else True,
    }}}
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(stt, "_get_provider", lambda cfg: provider)
    if case == "missing-ffmpeg":
        monkeypatch.setattr(audio, "_find_ffmpeg_binary", lambda: None)
    if case == "encode-failure":
        monkeypatch.setattr(audio, "_run_ffmpeg_stt_encode", Mock(side_effect=OSError("encoder unavailable")))
    seen = {}
    trim = audio._trim_silence_for_cloud_stt

    def inspect_trim(path, cfg):
        seen["trim_input"] = Path(path)
        if Path(path) != source:
            seen["loudness"] = audio._measure_integrated_loudness(path)
        return trim(path, cfg)

    monkeypatch.setattr(stt, "_trim_silence_for_cloud_stt", inspect_trim)

    def transcribe(path, model, **kwargs):
        seen["upload"] = Path(path)
        assert Path(path).is_file()
        seen["duration"] = audio._probe_audio_duration(path)
        if case == "dispatch-failure":
            raise RuntimeError("dispatch failure")
        return {"success": True, "transcript": "spoken words", "provider": provider}

    monkeypatch.setattr(stt, f"_transcribe_{provider}", transcribe)
    if case == "prepare-failure":
        def fail_prepare(path, work_dir, cfg):
            seen["work_dir"] = Path(work_dir)
            (Path(work_dir) / "partial.m4a").write_bytes(b"partial")
            raise RuntimeError("prepare failure")
        monkeypatch.setattr(stt, "_prepare_openai_audio_for_policy", fail_prepare, raising=False)
    if case in {"prepare-failure", "dispatch-failure"}:
        with pytest.raises(RuntimeError, match="failure"):
            stt.transcribe_audio(str(source))
    else:
        assert stt.transcribe_audio(str(source))["success"]
    assert source.read_bytes() == original
    if case == "prepare-failure":
        assert not seen["work_dir"].exists()
        return
    normalized = case in {"quiet", "dispatch-failure"}
    assert (seen["trim_input"] != source) == normalized
    if normalized:
        assert seen["loudness"] > -24
        assert 9 < seen["duration"] < 12  # both quiet spans survive, pause removed
    for path in (seen["trim_input"], seen["upload"]):
        if path != source:
            assert not path.parent.exists()


@pytest.mark.parametrize("primary,fallback,options,retry,recover", [
    ("few words", "a much more complete spoken request with all the words", {}, True, True),
    ("短い結果", "静かな録音でも完全な内容を最後まで正確に文字起こししました", {}, True, True),
    ("これは静かな録音ですが内容は最後まで正しく文字起こしされています", "unused", {}, False, False),
    ("one two three four five six seven eight nine ten eleven", "unused", {}, False, False),
    (None, "recovered words", {}, True, True),
    ("few words", None, {}, True, False),
    ("few words", "short", {}, True, False),
    ("few words", "unused", {"fallback_model": ""}, False, False),
    ("few words", "unused", {"fallback_model": "gpt-transcribe"}, False, False),
    ("few words", "unused", {"retry_min_words_per_second": 0}, False, False),
    ("few words", "unused", {"retry_min_words_per_second": True}, False, False),
    ("few words", "unused", {"retry_min_audio_seconds": 20}, False, False),
])
def test_dispatch_retries_only_when_needed_and_keeps_better_result(
    tmp_path, monkeypatch, primary, fallback, options, retry, recover,
):
    from tools import transcription_cloud as cloud

    cfg = {"openai": {"model": "gpt-transcribe", "language": "de",
                      "fallback_model": "gpt-4o-transcribe",
                      "retry_min_audio_seconds": 10, "retry_min_words_per_second": 1,
                      **options}, "prompt": "Names: Johannes"}
    results = [{"success": text is not None, "transcript": text or "", "provider": "openai"}
               for text in (primary, fallback)]
    backend = Mock(side_effect=results)
    monkeypatch.setattr(stt, "_apply_pre_transcription_hook",
                        lambda **kw: (kw["model"], "de", kw["prompt"]))
    monkeypatch.setattr(stt, "_transcribe_openai", backend)
    monkeypatch.setattr(cloud, "_probe_audio_duration", lambda path: 10.0, raising=False)
    path = str(tmp_path / "voice.wav")
    result = stt._dispatch_stt_provider(path, "openai", cfg)
    assert backend.call_count == (2 if retry else 1)
    assert result["transcript"] == (fallback if recover else primary or "")
    assert result.get("recovered_with_model") == ("gpt-4o-transcribe" if recover else None)
    for call in backend.call_args_list:
        assert call.args[0] == path
        assert call.kwargs == {"language": "de", "prompt": cfg["prompt"]}
    if retry:
        assert backend.call_args_list[1].args[1] == cfg["openai"]["fallback_model"]
