"""whisper.cpp runner.

Audio is converted to 16 kHz mono WAV with ffmpeg in a private temp dir (Telegram/WhatsApp voice
notes are Opus, which whisper.cpp cannot read), then decoded by ``whisper-cli`` (Metal on Apple
Silicon). Both subprocesses get a minimal environment — never Hermes's own, which holds API keys —
a fixed argv (no shell), and a timeout. The temp dir is deleted when the call returns, so no audio
outlives the transcription.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SEARCH_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"


class EngineError(RuntimeError):
    pass


def _which(name: str) -> Optional[str]:
    return shutil.which(name) or shutil.which(name, path=_SEARCH_PATH)


def whisper_cli() -> Optional[str]:
    return _which("whisper-cli")


def ffmpeg() -> Optional[str]:
    return _which("ffmpeg")


def _env() -> dict[str, str]:
    env = {"PATH": _SEARCH_PATH, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}
    tmp = os.environ.get("TMPDIR")
    if tmp:
        env["TMPDIR"] = tmp
    return env


_DETECTED_RE = re.compile(r"auto-detected language:\s*([a-z]{2,3})\s*\(p\s*=\s*([0-9.]+)\)")


@dataclass(frozen=True)
class Decoded:
    text: str
    language: Optional[str] = None      # only set when decoded with language="auto"
    probability: Optional[float] = None


@dataclass(frozen=True)
class DecodeOptions:
    model_path: Path
    language: str = "ar"
    vad_model_path: Optional[Path] = None
    threads: int = 4
    prompt: str = ""
    timeout_s: int = 600


def _run(argv: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv, env=_env(), stdin=subprocess.DEVNULL, capture_output=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EngineError(f"{Path(argv[0]).name} timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise EngineError(f"could not start {Path(argv[0]).name}: {exc}") from exc


def to_wav(src: str, work_dir: str, timeout_s: int = 120) -> str:
    ff = ffmpeg()
    if not ff:
        raise EngineError("ffmpeg not found (brew install ffmpeg)")
    dst = os.path.join(work_dir, "audio.wav")
    proc = _run([ff, "-nostdin", "-v", "error", "-y", "-i", src, "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", dst], timeout_s)
    if proc.returncode != 0 or not os.path.isfile(dst):
        raise EngineError("ffmpeg could not decode the audio: "
                          + proc.stderr.decode("utf-8", "replace").strip()[-300:])
    return dst


def transcribe(src: str, opts: DecodeOptions) -> Decoded:
    """Decode ``src``. With ``language="auto"`` Whisper detects the language in the same pass and
    the result carries it (parsed from whisper-cli's log; ``-np`` is not used for that reason)."""
    cli = whisper_cli()
    if not cli:
        raise EngineError("whisper-cli not found (brew install whisper-cpp)")
    if not opts.model_path.is_file():
        raise EngineError(f"model file missing: {opts.model_path.name} (run `hermes lahja setup`)")
    with tempfile.TemporaryDirectory(prefix="lahja-") as work:
        wav = to_wav(src, work)
        return decode_wav(wav, work, opts, cli)


def decode_wav(wav: str, work: str, opts: DecodeOptions, cli: Optional[str] = None) -> Decoded:
    cli = cli or whisper_cli()
    if not cli:
        raise EngineError("whisper-cli not found (brew install whisper-cpp)")
    out_base = os.path.join(work, "out")
    txt = out_base + ".txt"
    if os.path.exists(txt):
        os.unlink(txt)
    argv = [cli, "-m", str(opts.model_path), "-f", wav, "-l", opts.language or "auto",
            "-t", str(max(1, int(opts.threads))), "-nt", "-otxt", "-of", out_base]
    if opts.vad_model_path is not None:
        argv += ["--vad", "-vm", str(opts.vad_model_path)]
    if opts.prompt:
        argv += ["--prompt", opts.prompt]
    proc = _run(argv, opts.timeout_s)
    log = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise EngineError("whisper-cli failed: " + log.strip()[-300:])
    text = ""
    if os.path.isfile(txt):
        with open(txt, encoding="utf-8", errors="replace") as f:
            text = " ".join(line.strip() for line in f if line.strip())
    lang = prob = None
    m = _DETECTED_RE.search(log)
    if m:
        lang, prob = m.group(1), float(m.group(2))
    return Decoded(text=text, language=lang, probability=prob)


ARABIC = "ar"


@dataclass(frozen=True)
class Routed:
    decoded: Decoded
    language: Optional[str]          # the language the final text is in
    routed: bool                     # True when the other-languages model produced the text
    reason: str = ""                 # why it was routed: "detected" | "transliterated"


def transcribe_routed(src: str, primary: DecodeOptions, other_model: Optional[Path],
                      min_probability: float = 0.5) -> Routed:
    """Arabic → dialect model; anything else → ``other_model`` (stock Whisper).

    One ffmpeg conversion. The dialect model runs first with ``language="auto"``, so Arabic — the
    common case — costs a single pass: detection and transcription happen together. The audio is
    decoded again by ``other_model`` only when

    * Whisper detected another language with confidence >= ``min_probability`` (decoded in that
      language), or
    * Whisper said "Arabic" but the text is English written in Arabic letters — accented English
      is routinely detected as Arabic — (decoded as English; see :mod:`translit`).

    Without ``other_model`` the dialect model's own output is kept.
    """
    from . import translit

    cli = whisper_cli()
    if not cli:
        raise EngineError("whisper-cli not found (brew install whisper-cpp)")
    if not primary.model_path.is_file():
        raise EngineError(f"model file missing: {primary.model_path.name} (run `hermes lahja setup`)")
    with tempfile.TemporaryDirectory(prefix="lahja-") as work:
        wav = to_wav(src, work)
        first = decode_wav(wav, work, primary, cli)
        auto = primary.language == "auto"
        lang = first.language if auto else primary.language
        target, reason = None, ""
        if auto and first.text:
            if first.language not in (None, ARABIC) and (first.probability or 0.0) >= min_probability:
                target, reason = first.language, "detected"
            elif translit.looks_like_transliterated_english(first.text):
                target, reason = "en", "transliterated"
        if target is None or other_model is None:
            return Routed(first, lang, False)
        if not other_model.is_file():
            raise EngineError(f"other-languages model missing: {other_model}")
        second = decode_wav(wav, work, DecodeOptions(
            model_path=other_model, language=target, vad_model_path=primary.vad_model_path,
            threads=primary.threads, prompt=primary.prompt, timeout_s=primary.timeout_s), cli)
        return Routed(second, target, True, reason)
