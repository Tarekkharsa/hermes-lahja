"""Corrections the user teaches once: «جي إلي تي» => JLT.

Stored in the plugin's own profile-scoped state (``ctx.state``), as the two strings only — never
audio, never the transcript they came from. Applied after every transcription on whole-token
boundaries of the folded text, so «جي إلي تي» also matches «جي إلي تي،» but never the inside of a
longer word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .arabic import clean_spacing, fold_token

STATE_KEY = "corrections"
MAX_CORRECTIONS = 500
MAX_PHRASE_CHARS = 120
_SEPARATORS = ("=>", "->", "→", "⬅", "<-")


@dataclass(frozen=True)
class Correction:
    wrong: str
    right: str

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(t for t in (fold_token(x) for x in self.wrong.split()) if t)


def parse_rule(raw: str) -> Optional[Correction]:
    """Parse ``wrong => right`` (also ``->`` / ``→``). Returns None when malformed."""
    raw = (raw or "").strip()
    for sep in _SEPARATORS:
        if sep in raw:
            wrong, _, right = raw.partition(sep)
            if sep in ("⬅", "<-"):
                wrong, right = right, wrong
            wrong, right = wrong.strip().strip("\"'«»"), right.strip().strip("\"'«»")
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
    state.set(STATE_KEY, [{"wrong": c.wrong, "right": c.right} for c in items][:MAX_CORRECTIONS])


def add(state: Any, c: Correction) -> list[Correction]:
    items = [x for x in load(state) if x.key != c.key]
    items.append(c)
    save(state, items)
    return items


def remove(state: Any, index: int) -> Optional[Correction]:
    items = load(state)
    if not 1 <= index <= len(items):
        return None
    gone = items.pop(index - 1)
    save(state, items)
    return gone


_TRAIL_PUNCT_RE = re.compile(r"([،؛؟.,!?:]+)$")


def apply(text: str, items: list[Correction]) -> tuple[str, int]:
    """Replace every taught phrase; longest phrases first. Returns ``(text, replacements)``.

    Matching compares the folded tokens *joined without spaces*, over windows from one token to
    one more than the taught phrase, because Whisper splits spelled-out letters differently from
    one clip to the next («جي إل تي», «جي إلتي», «جي إلي تي»): teaching one covers its re-spacings.
    """
    if not text or not items:
        return text, 0
    tokens = text.split()
    count = 0
    for c in sorted(items, key=lambda x: len("".join(x.key)), reverse=True):
        target = "".join(c.key)
        sizes = range(len(c.key) + 1, 0, -1)  # widest first: n+1 … 1 tokens
        i, out = 0, []
        while i < len(tokens):
            hit = 0
            for n in sizes:
                window = tokens[i:i + n]
                if len(window) == n and "".join(fold_token(t) for t in window) == target:
                    hit = n
                    break
            if hit:
                m = _TRAIL_PUNCT_RE.search(tokens[i + hit - 1])
                out.append(c.right + (m.group(1) if m else ""))
                i += hit
                count += 1
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return clean_spacing(" ".join(tokens)), count
