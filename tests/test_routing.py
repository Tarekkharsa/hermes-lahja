"""Language routing: Arabic → dialect model in one pass, other languages → other_languages_model."""

from pathlib import Path

import pytest


@pytest.fixture
def eng(lahja, monkeypatch, tmp_path):
    import hermes_plugins.lahja.engine as engine
    calls = []
    dialect = tmp_path / "dialect.bin"
    stock = tmp_path / "stock.bin"
    dialect.write_bytes(b"x")
    stock.write_bytes(b"x")
    script = {}

    def fake_decode(wav, work, opts, cli=None):
        calls.append((Path(opts.model_path).name, opts.language))
        return script[(Path(opts.model_path).name, opts.language)]

    monkeypatch.setattr(engine, "whisper_cli", lambda: "/bin/whisper-cli")
    monkeypatch.setattr(engine, "to_wav", lambda src, work, timeout_s=120: work + "/audio.wav")
    monkeypatch.setattr(engine, "decode_wav", fake_decode)
    return engine, calls, script, dialect, stock


def test_arabic_is_one_pass(eng):
    engine, calls, script, dialect, stock = eng
    script[("dialect.bin", "auto")] = engine.Decoded("هلأ رايح", "ar", 0.99)
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "auto"), stock)
    assert (r.decoded.text, r.language, r.routed) == ("هلأ رايح", "ar", False)
    assert calls == [("dialect.bin", "auto")]


def test_english_goes_to_stock_model(eng):
    engine, calls, script, dialect, stock = eng
    script[("dialect.bin", "auto")] = engine.Decoded("ه، can you book", "en", 0.91)
    script[("stock.bin", "en")] = engine.Decoded("Hey, can you book", None, None)
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "auto"), stock)
    assert (r.decoded.text, r.language, r.routed) == ("Hey, can you book", "en", True)
    assert calls == [("dialect.bin", "auto"), ("stock.bin", "en")]


def test_low_confidence_foreign_stays_on_dialect(eng):
    engine, calls, script, dialect, stock = eng
    script[("dialect.bin", "auto")] = engine.Decoded("بدي أعمل pull request", "en", 0.3)
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "auto"), stock)
    assert r.routed is False and len(calls) == 1


def test_no_other_model_keeps_dialect_output(eng):
    engine, calls, script, dialect, _ = eng
    script[("dialect.bin", "auto")] = engine.Decoded("can you book", "en", 0.95)
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "auto"), None)
    assert (r.decoded.text, r.language, r.routed) == ("can you book", "en", False)


def test_silence_never_triggers_second_pass(eng):
    engine, calls, script, dialect, stock = eng
    script[("dialect.bin", "auto")] = engine.Decoded("", None, None)
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "auto"), stock)
    assert r.decoded.text == "" and len(calls) == 1


def test_forced_arabic_skips_detection(eng):
    engine, calls, script, dialect, stock = eng
    script[("dialect.bin", "ar")] = engine.Decoded("مرحبا")
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "ar"), stock)
    assert (r.language, r.routed) == ("ar", False)


def test_missing_other_model_keeps_first_result(eng, tmp_path):
    engine, calls, script, dialect, _ = eng
    script[("dialect.bin", "auto")] = engine.Decoded("hello", "en", 0.99)
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "auto"), tmp_path / "nope.bin")
    assert (r.decoded.text, r.routed, r.reason) == ("hello", False, "other_model_missing")


def test_detected_language_is_parsed_from_log(lahja):
    import hermes_plugins.lahja.engine as engine
    m = engine._DETECTED_RE.search("whisper_full_with_state: auto-detected language: en (p = 0.911849)\n")
    assert (m.group(1), float(m.group(2))) == ("en", 0.911849)


@pytest.mark.parametrize("value,ok", [
    ("", True), ("turbo-stock-q5", True), ("/Users/x/.cache/whisper/ggml-large-v3.bin", True),
    ("~/.cache/whisper/ggml-large-v3.bin", True), ("relative/model.bin", False), ("/etc/passwd", False),
])
def test_other_languages_model_setting(lahja, tmp_path, value, ok):
    from conftest import FakeCtx
    import hermes_plugins.lahja.engine as engine
    p = lahja.LahjaProvider(FakeCtx(tmp_path, {"other_languages_model": value}))
    if ok:
        path, label = p._other_model()
        assert (path is None) == (value == "")
    else:
        with pytest.raises(engine.EngineError):
            p._other_model()


def test_provider_reports_routing(lahja, tmp_path, monkeypatch):
    from conftest import FakeCtx
    import hermes_plugins.lahja.engine as engine
    import hermes_plugins.lahja.models as models
    c = FakeCtx(tmp_path, {"other_languages_model": "turbo-stock-q5"})
    for m in (models.VAD_MODEL, models.MODELS[models.DEFAULT_MODEL]):
        models.models_dir(tmp_path); models.path_for(tmp_path, m).write_bytes(b"x")
    seen = {}

    def fake(src, opts, other, **k):
        seen["lang"], seen["other"] = opts.language, other
        return engine.Routed(engine.Decoded("Hey, can you book"), "en", True)

    monkeypatch.setattr(engine, "transcribe_routed", fake)
    r = lahja.LahjaProvider(c).transcribe("/a.ogg")
    assert seen["lang"] == "auto" and seen["other"].name == "ggml-large-v3-turbo-q5_0.bin"
    assert (r["transcript"], r["language"], r["model"], r["routed"]) == (
        "Hey, can you book", "en", "turbo-stock-q5", "other_languages_model")


# -- regression: 2026-11-06 09:03, a Levantine voice note came back translated into English ----
# Hermes passed language="en" (stt.language: en in config.yaml, the template default) and the
# plugin forced large-v3 to English, which makes Whisper translate.
@pytest.mark.parametrize("setting,hint,expected", [
    ("auto", "en", "auto"),      # the bug: Hermes's default "en" hint must not force English
    ("auto", "en-US", "auto"),
    ("auto", "fr", "auto"),
    ("auto", None, "auto"),
    ("auto", "ar", "ar"),
    ("auto", "ar-AE", "ar"),
    ("ar", "en", "ar"),          # the plugin's own "ar" setting wins over any hint
    ("en", None, "auto"),        # an invalid non-Arabic setting falls back to detection
    ("", "", "auto"),
])
def test_language_never_forced_to_non_arabic(lahja, tmp_path, setting, hint, expected):
    from conftest import FakeCtx
    p = lahja.LahjaProvider(FakeCtx(tmp_path, {"language": setting}))
    assert p._language(hint) == expected


def test_english_hint_with_arabic_speech_stays_arabic(lahja, tmp_path, monkeypatch):
    from conftest import FakeCtx
    import hermes_plugins.lahja.engine as engine
    import hermes_plugins.lahja.models as models
    c = FakeCtx(tmp_path, {"other_languages_model": "turbo-stock-q5"})
    for m in (models.VAD_MODEL, models.MODELS[models.DEFAULT_MODEL], models.MODELS["turbo-stock-q5"]):
        models.models_dir(tmp_path); models.path_for(tmp_path, m).write_bytes(b"x")
    calls = []

    def fake_decode(wav, work, opts, cli=None):
        calls.append((Path(opts.model_path).name, opts.language))
        return engine.Decoded("شو عم يصير؟ هلأ أنا عم بيبني ال بلغين", "ar", 0.99)

    monkeypatch.setattr(engine, "whisper_cli", lambda: "/bin/whisper-cli")
    monkeypatch.setattr(engine, "to_wav", lambda src, work, timeout_s=120: work + "/audio.wav")
    monkeypatch.setattr(engine, "decode_wav", fake_decode)
    r = lahja.LahjaProvider(c).transcribe("/a.ogg", language="en")
    assert r["transcript"].startswith("شو عم يصير")
    assert r["language"] == "ar" and "routed" not in r
    assert calls == [("ggml-whisper-large-v3-turbo-arabic-dialectal-v2-q5_0.bin", "auto")]
    assert all(lang != "en" for _, lang in calls)


# -- accented English detected as Arabic (real voice note, 2026-11-06 12:34) --------------------
TRANSLIT = ("ثن، وي سند أنثر إيميل، أند، أه، ذاتس لايك يو غايز ذي رسبونسبل فور ذس، أند يو كان چك ذات، "
            "أند ذن شي ستارت تراينغ إن ذي أوفيس. آي دونت نو. ذاتس فري باد.")


def test_transliterated_english_is_redecoded_as_english(eng):
    engine, calls, script, dialect, stock = eng
    script[("dialect.bin", "auto")] = engine.Decoded(TRANSLIT, "ar", 0.99995)
    script[("stock.bin", "en")] = engine.Decoded("Then we send another email and that's like you guys")
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "auto"), stock)
    assert (r.decoded.text[:9], r.language, r.routed, r.reason) == ("Then we s", "en", True, "transliterated")
    assert calls == [("dialect.bin", "auto"), ("stock.bin", "en")]


def test_forced_arabic_never_rechecks_script(eng):
    engine, calls, script, dialect, stock = eng
    script[("dialect.bin", "ar")] = engine.Decoded(TRANSLIT)
    r = engine.transcribe_routed("/a.ogg", engine.DecodeOptions(dialect, "ar"), stock)
    assert r.routed is False and len(calls) == 1


def test_translit_scores(lahja):
    from hermes_plugins.lahja import translit
    assert translit.looks_like_transliterated_english(TRANSLIT)
    for arabic in (
        "شو عم يصير؟ هلأ أنا عم بيبني ال بلغين، في ايجنت تاني عم يشتر عليها، وعم يجرب ال آربك سبيتش تو تكست.",
        "وانا كانت عندي فكرة بس ما كان في وقت، إي والله شي بيجنن، تو وصلت على البيت وانت وين؟",
        "ما بعرف، حاسس إنه هي القصة كلها مالها طعمة، ومانا مفيدة",
        "بدي أعمل full request على الريبو تبع الشغل، أوكي؟ لايك ما في شي",
        "ذا الشي اللي قلتلك عنه",           # Gulf «ذا» (this) alone must not trip it
    ):
        assert not translit.looks_like_transliterated_english(arabic), arabic
