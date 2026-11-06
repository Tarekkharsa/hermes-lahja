"""Corrections the user teaches once: «جي إلي تي» => JLT.

Stored in the plugin's own profile-scoped state (``ctx.state``), as the two strings only — never
audio, never the transcript they came from. Applied after every transcription:

* on whole words of the folded text (diacritics, hamza seats, punctuation ignored);
* a rule made of *spelled-out letters* («جي إلي تي», "G L T": every word 1–3 letters) also matches
  the same letters spaced differently («جيإلي تي», «جيإليتي»), because Whisper splits spelled
  letters differently from clip to clip. Any other rule matches exactly its own words, so «مالك»
  never swallows «ما لك»;
* each word is replaced at most once: rules never chain (A→B then B→C does not turn A into C);
* quotes and brackets around the replaced words are kept.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .arabic import clean_spacing, fold_token

STATE_KEY = "corrections"
MAX_CORRECTIONS = 500
MAX_PHRASE_CHARS = 120
_SEPARATORS = ("=>", "->", "→", "⬅", "<-")
_SPELLED_MAX_LETTERS = 3
_JOINERS = (chr(0x200C), chr(0x200D))  # ZWNJ/ZWJ are legitimate inside Persian/Arabic words
_LOCK = threading.Lock()  # read-modify-write of the list (Hermes state has no atomic update)


class CorrectionLimit(Exception):
    pass


@dataclass(frozen=True)
class Correction:
    wrong: str
    right: str

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(t for t in (fold_token(x) for x in self.wrong.split()) if t)

    @property
    def spelled(self) -> bool:
        """Spelled-out letters/initialisms: every word is 1–3 letters (جي إلي تي, G L T, JLT)."""
        return all(len(t) <= _SPELLED_MAX_LETTERS for t in self.key)


def _clean_phrase(s: str) -> str:
    s = s.strip().strip("\"'«»")
    if any(unicodedata.category(ch) in ("Cc", "Cf") and ch not in _JOINERS for ch in s):
        return ""  # no control / invisible characters: a rule may not smuggle hidden text
    return " ".join(s.split())


def parse_rule(raw: str) -> Optional[Correction]:
    """Parse ``wrong => right`` (also ``->`` / ``→``). Returns None when malformed."""
    raw = (raw or "").strip()
    for sep in _SEPARATORS:
        if sep in raw:
            wrong, _, right = raw.partition(sep)
            if sep in ("⬅", "<-"):
                wrong, right = right, wrong
            wrong, right = _clean_phrase(wrong), _clean_phrase(right)
            if not wrong or not right or len(wrong) > MAX_PHRASE_CHARS or len(right) > MAX_PHRASE_CHARS:
                return None
            c = Correction(wrong, right)
            return c if c.key else None
    return None


def load(state: Any) -> list[Correction]:
    raw = state.get(STATE_KEY, []) if state is not None else []
    out: list[Correction] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("wrong"), str) and isinstance(item.get("right"), str):
            c = Correction(item["wrong"], item["right"])
            if c.key:
                out.append(c)
    return out


def save(state: Any, items: Iterable[Correction]) -> None:
    state.set(STATE_KEY, [{"wrong": c.wrong, "right": c.right} for c in items])


def add(state: Any, c: Correction) -> list[Correction]:
    """Add or replace ``c`` (same folded key replaces). Raises CorrectionLimit at the cap."""
    with _LOCK:
        current = load(state)
        items = [x for x in current if x.key != c.key]
        if len(items) >= MAX_CORRECTIONS:
            raise CorrectionLimit(
                f"Lahja keeps at most {MAX_CORRECTIONS} corrections; remove one first (/stt-fix list).")
        items.append(c)
        save(state, items)
        return items


def remove(state: Any, index: int) -> Optional[Correction]:
    with _LOCK:
        items = load(state)
        if not 1 <= index <= len(items):
            return None
        gone = items.pop(index - 1)
        save(state, items)
        return gone


_LEAD_RE = re.compile(r"^([\"'«(\[{]+)")
_TRAIL_RE = re.compile(r"([\"'»)\]}،؛؟.,!?:]+)$")


def apply(text: str, items: list[Correction]) -> tuple[str, int]:
    """Replace every taught phrase. Returns ``(text, replacements)``.

    One left-to-right pass with dictionary lookups (500 rules over a long note stays fast). At each
    word the widest matching window wins, so a longer phrase beats a shorter one starting at the
    same word; replaced words are never looked at again, so rules cannot chain.
    """
    if not text or not items:
        return text, 0
    exact: dict[tuple[str, ...], Correction] = {}
    spelled: dict[str, Correction] = {}
    for c in items:
        exact[c.key] = c
        if c.spelled:
            spelled["".join(c.key)] = c
    widest = max(len(c.key) for c in items) + 1
    tokens = text.split()
    folds = [fold_token(t) for t in tokens]
    out: list[str] = []
    count = 0
    i = 0
    while i < len(tokens):
        hit: Optional[tuple[Correction, int]] = None
        for size in range(min(widest, len(tokens) - i), 0, -1):
            window = folds[i:i + size]
            if not all(window):
                continue
            c = exact.get(tuple(window))
            if c is None:
                c = spelled.get("".join(window))
                if c is not None and size > len(c.key) + 1:
                    c = None
            if c is not None:
                hit = (c, size)
                break
        if hit is None:
            out.append(tokens[i])
            i += 1
            continue
        c, size = hit
        lead = _LEAD_RE.search(tokens[i])
        trail = _TRAIL_RE.search(tokens[i + size - 1])
        out.append((lead.group(1) if lead else "") + c.right + (trail.group(1) if trail else ""))
        i += size
        count += 1
    return clean_spacing(" ".join(out)), count
