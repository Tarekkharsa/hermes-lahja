import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def mods(lahja):
    import hermes_plugins.lahja.engine as engine
    import hermes_plugins.lahja.models as models
    return engine, models


def _install_fake_models(models, data_dir: Path):
    for m in (models.VAD_MODEL, models.MODELS[models.DEFAULT_MODEL]):
        models.models_dir(data_dir)
        models.path_for(data_dir, m).write_bytes(b"x")


def test_register(lahja, ctx):
    lahja.register(ctx)
    assert ctx.registered["provider"].name == "lahja"
    assert "/stt-fix" in ctx.registered
    assert "cli:lahja" in ctx.registered


def test_name_is_not_a_builtin(lahja, ctx):
    assert lahja.LahjaProvider(ctx).name not in {
        "local", "local_command", "groq", "openai", "mistral", "xai", "elevenlabs", "deepinfra"}


def test_pipeline_filters_and_corrects(lahja, ctx, mods, monkeypatch):
    engine, models = mods
    _install_fake_models(models, ctx.state.data_dir)
    seen = {}

    def fake_routed(src, opts, other, min_probability=0.5, **kw):
        seen["opts"] = opts
        return engine.Routed(engine.Decoded("بدي روح على جي إلي تي بكرا ترجمة نانسي قنقر", "ar", 1.0), "ar", False)

    monkeypatch.setattr(engine, "transcribe_routed", fake_routed)
    fix = lahja.make_stt_fix(lambda: ctx.state)
    assert fix("جي إلي تي => JLT").startswith("Learned")
    r = lahja.LahjaProvider(ctx).transcribe("/x.ogg", language="ar-AE", prompt="JLT")
    assert r["success"] is True
    assert r["transcript"] == "بدي روح على JLT بكرا"
    assert r["hallucinations_removed"] == 1 and r["corrections_applied"] == 1
    assert seen["opts"].language == "ar"  # an explicit caller hint wins over the auto default
    assert seen["opts"].vad_model_path is not None
    assert seen["opts"].prompt == "JLT"


def test_silence_becomes_empty_success(lahja, ctx, mods, monkeypatch):
    engine, models = mods
    _install_fake_models(models, ctx.state.data_dir)
    monkeypatch.setattr(engine, "transcribe_routed",
                        lambda src, opts, other, **k: engine.Routed(engine.Decoded("ترجمة نانسي قنقر"), "ar", False))
    r = lahja.LahjaProvider(ctx).transcribe("/x.wav")
    assert r["success"] is True and r["transcript"] == ""


def test_engine_errors_become_envelope(lahja, ctx, mods, monkeypatch):
    engine, _ = mods

    def boom(src, opts, other, **k):
        raise engine.EngineError("whisper-cli not found")

    monkeypatch.setattr(engine, "transcribe_routed", boom)
    r = lahja.LahjaProvider(ctx).transcribe("/x.wav")
    assert r == {"success": False, "transcript": "", "provider": "lahja",
                 "error": "lahja: whisper-cli not found"}


def test_unexpected_errors_never_raise(lahja, ctx, mods, monkeypatch):
    engine, _ = mods
    monkeypatch.setattr(engine, "transcribe_routed", lambda s, o, x, **k: 1 / 0)
    assert lahja.LahjaProvider(ctx).transcribe("/x.wav")["success"] is False


def test_filter_can_be_disabled(lahja, tmp_path, mods, monkeypatch):
    from conftest import FakeCtx
    engine, models = mods
    c = FakeCtx(tmp_path, {"hallucination_filter": False, "vad": False})
    _install_fake_models(models, tmp_path)
    seen = {}

    def fake(src, opts, other, **k):
        seen["vad"] = opts.vad_model_path
        return engine.Routed(engine.Decoded("ترجمة نانسي قنقر"), "ar", False)

    monkeypatch.setattr(engine, "transcribe_routed", fake)
    r = lahja.LahjaProvider(c).transcribe("/x.wav")
    assert r["transcript"] == "ترجمة نانسي قنقر" and seen["vad"] is None


def test_subprocess_env_is_minimal(mods, monkeypatch):
    engine, _ = mods
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    env = engine._env()
    assert "OPENROUTER_API_KEY" not in env
    assert set(env) <= {"PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH"}


def test_subprocess_env_allowlist_on_linux_runners(mods, monkeypatch):
    engine, _ = mods
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "nope")
    env = engine._env()
    assert env["LD_LIBRARY_PATH"] == "/opt/cuda/lib" and "AWS_SECRET_ACCESS_KEY" not in env


def test_stt_fix_command(lahja, ctx):
    fix = lahja.make_stt_fix(lambda: ctx.state)
    assert "No corrections yet" in fix("list")
    assert "Couldn't read" in fix("JLT")
    fix("GLT -> JLT")
    assert "1. GLT → JLT" in fix("list")
    assert "Removed" in fix("remove 1")
    assert "No such correction" in fix("remove x")


def test_download_rejects_bad_checksum(mods, tmp_path, monkeypatch):
    _, models = mods
    import io

    class Resp(io.BytesIO):
        headers = {"Content-Length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda req, timeout: Resp(b"abc"))
    with pytest.raises(ValueError, match="checksum"):
        models.download(tmp_path, models.VAD_MODEL)
    assert not any(p.name.startswith(".download-") for p in models.models_dir(tmp_path).iterdir())
    assert not models.is_installed(tmp_path, models.VAD_MODEL)


# -- real whisper.cpp run (opt-in: LAHJA_TEST_MODELS=<dir with the two .bin files>) -----------
MODELS_DIR = os.environ.get("LAHJA_TEST_MODELS")
needs_real = pytest.mark.skipif(
    not (MODELS_DIR and shutil.which("whisper-cli") and shutil.which("ffmpeg")),
    reason="set LAHJA_TEST_MODELS and install whisper-cpp + ffmpeg")


@needs_real
def test_real_silence_is_empty(lahja, tmp_path, mods):
    import subprocess
    engine, models = mods
    from conftest import FakeCtx
    c = FakeCtx(tmp_path)
    md = models.models_dir(tmp_path)
    for m in (models.VAD_MODEL, models.MODELS[models.DEFAULT_MODEL]):
        os.symlink(Path(MODELS_DIR) / m.filename, md / m.filename)
    wav = tmp_path / "silence.ogg"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                    "-t", "8", "-c:a", "libopus", str(wav)], check=True)
    r = lahja.LahjaProvider(c).transcribe(str(wav))
    assert r["success"] is True and r["transcript"] == ""
