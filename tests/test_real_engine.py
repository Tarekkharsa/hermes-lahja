"""Real whisper.cpp runs (no mocks). Opt-in: set LAHJA_REAL_MODEL (any ggml Whisper .bin, e.g.
ggml-tiny.bin), LAHJA_REAL_VAD (ggml-silero-v5.1.2.bin) and LAHJA_REAL_SAMPLE (an Arabic .wav).
CI runs this on macOS with the tiny model; locally any model works."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

MODEL = os.environ.get("LAHJA_REAL_MODEL")
VAD = os.environ.get("LAHJA_REAL_VAD")
SAMPLE = os.environ.get("LAHJA_REAL_SAMPLE")
pytestmark = pytest.mark.skipif(
    not (MODEL and VAD and SAMPLE and shutil.which("whisper-cli") and shutil.which("ffmpeg")),
    reason="set LAHJA_REAL_MODEL, LAHJA_REAL_VAD, LAHJA_REAL_SAMPLE; needs whisper-cli and ffmpeg")


@pytest.fixture
def engine(lahja):
    import hermes_plugins.lahja.engine as e
    return e


def _opts(engine, language="auto"):
    return engine.DecodeOptions(model_path=Path(MODEL), language=language, vad_model_path=Path(VAD))


def test_detects_arabic_from_real_whisper_log(engine):
    r = engine.transcribe_routed(SAMPLE, _opts(engine), None, timeout_s=300)
    assert r.language == "ar"               # parsed from whisper-cli's "auto-detected language" line
    assert r.decoded.probability and r.decoded.probability > 0.5
    assert r.decoded.text                    # some Arabic text came back
    assert any("؀" <= ch <= "ۿ" for ch in r.decoded.text)


def test_opus_voice_note_is_converted(engine, tmp_path):
    ogg = tmp_path / "note.ogg"             # Telegram/WhatsApp voice notes are Opus
    subprocess.run(["ffmpeg", "-v", "error", "-i", SAMPLE, "-c:a", "libopus", str(ogg)], check=True)
    r = engine.transcribe_routed(str(ogg), _opts(engine, "ar"), None, timeout_s=300)
    assert r.decoded.text and r.language == "ar"


@pytest.mark.parametrize("src", ["anullsrc=r=16000:cl=mono", "anoisesrc=r=16000:color=pink:amplitude=0.02"])
def test_vad_returns_nothing_for_non_speech(engine, tmp_path, src):
    wav = tmp_path / "quiet.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", src, "-t", "8", str(wav)], check=True)
    r = engine.transcribe_routed(str(wav), _opts(engine, "ar"), None, timeout_s=300)
    assert r.decoded.text == ""


def test_whisper_never_sees_hermes_secrets(engine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-must-not-leak")
    seen = {}
    real_run = subprocess.run

    def spy(argv, **kw):
        seen.setdefault("envs", []).append(kw.get("env") or {})
        return real_run(argv, **kw)

    monkeypatch.setattr(engine.subprocess, "run", spy)
    engine.transcribe_routed(SAMPLE, _opts(engine, "ar"), None, timeout_s=300)
    assert seen["envs"] and all("OPENROUTER_API_KEY" not in e for e in seen["envs"])
