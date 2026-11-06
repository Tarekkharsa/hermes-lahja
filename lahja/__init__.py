"""Lahja (لهجة): local Arabic-dialect speech-to-text for Hermes.

    voice note → ffmpeg 16 kHz → Silero VAD → dialect Whisper (whisper.cpp) →
    hallucination filter → your taught corrections → Hermes

Select it with ``stt.provider: lahja``. Privacy: audio is decoded locally and deleted with the
call's temp dir; the plugin stores only the model files and the corrections you teach.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

from . import corrections, engine, filters, models

logger = logging.getLogger(__name__)

PROVIDER_NAME = "lahja"
PLUGIN_ID = "lahja"


def _bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


class LahjaProvider(TranscriptionProvider):
    def __init__(self, ctx: Any):
        self._ctx = ctx

    # -- identity ---------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Lahja — Arabic dialects (local)"

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "Arabic dialect Whisper on whisper.cpp, VAD, hallucination filter, taught corrections",
            "env_vars": [],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": m.id, "display": m.display} for m in models.MODELS.values()]

    def default_model(self) -> Optional[str]:
        return models.DEFAULT_MODEL

    # -- settings ---------------------------------------------------------------------------
    def _cfg(self, key: str, default: Any) -> Any:
        try:
            return self._ctx.get_config(key, default)
        except Exception:
            return default

    def _data_dir(self):
        return self._ctx.state.data_dir

    def _model(self, override: Optional[str]) -> models.ModelFile:
        mid = (override or self._cfg("stt_model", models.DEFAULT_MODEL) or models.DEFAULT_MODEL).strip()
        return models.MODELS.get(mid) or models.MODELS[models.DEFAULT_MODEL]

    def _language(self, hint: Optional[str]) -> str:
        """``"ar"`` or ``"auto"`` — never another language.

        Forcing Whisper to a language the speaker is not using makes it *translate* into that
        language. Hermes passes a language hint from ``stt.language`` / ``HERMES_LOCAL_STT_LANGUAGE``,
        and config templates default it to ``"en"``, so obeying a non-Arabic hint would turn every
        Arabic voice note into English. A hint of ``ar`` (from the caller or this plugin's
        ``language`` setting) forces Arabic; anything else means detect per voice note, and
        non-Arabic speech is then decoded in its own detected language.
        """
        for candidate in (self._cfg("language", "auto"), hint):
            code = str(candidate or "").strip().split("-")[0].split("_")[0].lower()
            if code == engine.ARABIC:
                return engine.ARABIC
        return "auto"

    def _other_model(self) -> tuple[Optional[Path], str]:
        """Resolve ``other_languages_model``: '' → none, a catalog id → its downloaded file, or an
        absolute path to a ggml .bin you already have (e.g. the large-v3 your old setup used)."""
        raw = str(self._cfg("other_languages_model", "") or "").strip()
        if not raw:
            return None, ""
        if raw in models.MODELS:
            return models.path_for(self._data_dir(), models.MODELS[raw]), raw
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute() or p.suffix != ".bin":
            raise engine.EngineError(
                f"other_languages_model must be a model id or an absolute path to a ggml .bin: {raw!r}")
        return p, p.name

    def is_available(self) -> bool:
        try:
            return bool(engine.whisper_cli() and engine.ffmpeg()
                        and models.is_installed(self._data_dir(), self._model(None)))
        except Exception as exc:
            logger.warning("lahja availability check failed: %s", exc)
            return False

    # -- transcription ----------------------------------------------------------------------
    def _error(self, message: str) -> Dict[str, Any]:
        return {"success": False, "transcript": "", "provider": PROVIDER_NAME, "error": message}

    def transcribe(self, file_path: str, *, model: Optional[str] = None,
                   language: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
        try:
            m = self._model(model)
            data_dir = self._data_dir()
            use_vad = _bool(self._cfg("vad", True), True)
            vad_path = models.path_for(data_dir, models.VAD_MODEL) if use_vad else None
            if vad_path is not None and not vad_path.is_file():
                logger.warning("lahja: VAD model missing, decoding without VAD (run `hermes lahja setup`)")
                vad_path = None
            lang = self._language(language)
            prompt = extra.get("prompt") if isinstance(extra.get("prompt"), str) else ""
            other_path, other_id = self._other_model()
            primary = engine.DecodeOptions(
                model_path=models.path_for(data_dir, m),
                language=lang,
                vad_model_path=vad_path,
                threads=_int(self._cfg("threads", 4), 4, 1, 32),
                prompt=prompt[:400],
            )
            out = engine.transcribe_routed(file_path, primary, other_path)
        except engine.EngineError as exc:
            return self._error(f"lahja: {exc}")
        except Exception as exc:  # the ABC forbids raising
            logger.error("lahja: unexpected failure: %s", exc, exc_info=True)
            return self._error(f"lahja: transcription failed: {exc}")

        raw = out.decoded.text
        text, removed = raw, []
        if _bool(self._cfg("hallucination_filter", True), True):
            text, removed = filters.strip_hallucinations(raw)
        try:
            text, fixed = corrections.apply(text, corrections.load(self._ctx.state))
        except Exception as exc:
            logger.warning("lahja: corrections skipped: %s", exc)
            fixed = 0
        result: Dict[str, Any] = {
            "success": True, "transcript": text, "provider": PROVIDER_NAME,
            "model": other_id if out.routed else m.id, "vad": vad_path is not None,
        }
        if out.language:
            result["language"] = out.language
        if out.routed:
            result["routed"] = "other_languages_model"
            result["route_reason"] = out.reason
        if removed:
            result["hallucinations_removed"] = len(removed)
        if fixed:
            result["corrections_applied"] = fixed
        return result


# -- /stt-fix ---------------------------------------------------------------------------------
_HELP = (
    "Teach Lahja a correction once; it applies to every later voice note.\n"
    "  /stt-fix جي إلي تي => JLT     add (also -> or →)\n"
    "  /stt-fix list                 show taught corrections\n"
    "  /stt-fix remove 2             delete correction #2\n"
    "Only the two phrases are stored, never audio or transcripts."
)


def _format_list(items: List[corrections.Correction]) -> str:
    if not items:
        return "No corrections yet.\n\n" + _HELP
    lines = [f"{i}. {c.wrong} → {c.right}" for i, c in enumerate(items, 1)]
    return "Lahja corrections:\n" + "\n".join(lines)


def make_stt_fix(get_state):
    """``get_state`` is called per invocation: the profile-scoped state is resolved when the
    command runs, never at register() time."""
    def stt_fix(raw_args: str = "", **_: Any) -> str:
        state = get_state()
        arg = (raw_args or "").strip()
        if not arg or arg in {"help", "-h", "--help"}:
            return _HELP + "\n\n" + _format_list(corrections.load(state))
        if arg in {"list", "ls"}:
            return _format_list(corrections.load(state))
        head, _, rest = arg.partition(" ")
        if head in {"remove", "rm", "delete"}:
            try:
                gone = corrections.remove(state, int(rest.strip()))
            except ValueError:
                gone = None
            return f"Removed: {gone.wrong} → {gone.right}" if gone else "No such correction. Try /stt-fix list."
        rule = corrections.parse_rule(arg)
        if rule is None:
            return "Couldn't read that. Use: /stt-fix <what it wrote> => <what you said>\n\n" + _HELP
        items = corrections.add(state, rule)
        return f"Learned: {rule.wrong} → {rule.right} ({len(items)} total). Applies from the next voice note."
    return stt_fix


# -- `hermes lahja ...` -----------------------------------------------------------------------
def _cli_setup(subparser) -> None:
    sp = subparser.add_subparsers(dest="lahja_cmd")
    s = sp.add_parser("setup", help="Download the dialect model and VAD (checksum-verified)")
    s.add_argument("--model", default=None, help="model id (see `hermes lahja models`)")
    sp.add_parser("models", help="List available models and what is installed")
    sp.add_parser("status", help="Check whisper-cli, ffmpeg and model files")
    t = sp.add_parser("test", help="Transcribe one audio file with Lahja")
    t.add_argument("file")
    t.add_argument("--model", default=None)


def make_cli_handler(ctx: Any):
    def handler(args) -> int:
        data_dir = ctx.state.data_dir
        cmd = getattr(args, "lahja_cmd", None)
        if cmd == "models":
            for m in [*models.MODELS.values(), models.VAD_MODEL]:
                mark = "installed" if models.is_installed(data_dir, m) else "-"
                print(f"{m.id:26s} {mark:9s} {m.display}  [{m.license}; {m.source}]")
            return 0
        if cmd == "setup":
            mid = getattr(args, "model", None) or ctx.get_config("stt_model", models.DEFAULT_MODEL)
            m = models.MODELS.get(mid)
            if m is None:
                print(f"Unknown model {mid!r}; see `hermes lahja models`.")
                return 2
            for f in (models.VAD_MODEL, m):
                print(f"Downloading {f.display} …")
                last = [-1]

                def progress(done: int, total: int, last=last) -> None:
                    pct = int(done * 100 / total) if total else 0
                    if pct >= last[0] + 10:
                        last[0] = pct
                        print(f"  {pct}%", flush=True)
                path = models.download(data_dir, f, progress)
                print(f"  ok, sha256 verified → {path}")
            print("Done. Set `stt.provider: lahja` in config.yaml to use it.")
            return 0
        if cmd == "test":
            r = LahjaProvider(ctx).transcribe(args.file, model=getattr(args, "model", None))
            print(r.get("transcript") if r.get("success") else f"error: {r.get('error')}")
            extras = {k: v for k, v in r.items() if k not in {"transcript", "success", "provider"}}
            print(extras)
            return 0 if r.get("success") else 1
        # status (default)
        ok = True
        for label, found in (("whisper-cli", engine.whisper_cli()), ("ffmpeg", engine.ffmpeg())):
            print(f"{label:12s} {found or 'MISSING'}")
            ok &= bool(found)
        for m in (models.VAD_MODEL, LahjaProvider(ctx)._model(None)):
            inst = models.is_installed(data_dir, m)
            print(f"{m.id:12s} {'installed' if inst else 'MISSING (hermes lahja setup)'}")
            ok &= inst
        try:
            other, other_id = LahjaProvider(ctx)._other_model()
        except engine.EngineError as exc:
            other, other_id = None, f"INVALID: {exc}"
            ok = False
        if other is None:
            print(f"other langs  {other_id or 'not set (non-Arabic stays on the dialect model)'}")
        else:
            print(f"other langs  {other_id} {'found' if other.is_file() else 'MISSING'}")
            ok &= other.is_file()
        print(f"corrections  {len(corrections.load(ctx.state))}")
        return 0 if ok else 1
    return handler


def register(ctx) -> None:
    ctx.register_transcription_provider(LahjaProvider(ctx))
    ctx.register_command(
        "stt-fix", make_stt_fix(lambda: ctx.state),
        description="Teach Lahja a speech-to-text correction (e.g. جي إلي تي => JLT)",
        args_hint="<heard> => <meant> | list | remove N",
    )
    ctx.register_cli_command(
        "lahja", help="Lahja Arabic speech-to-text: setup, models, status, test",
        setup_fn=_cli_setup, handler_fn=make_cli_handler(ctx),
        description="Download models and test the Lahja Arabic dialect STT provider.",
    )
