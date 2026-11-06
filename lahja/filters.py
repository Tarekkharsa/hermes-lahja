"""Known Whisper hallucinations in Arabic.

Whisper was trained on subtitled video, so on silence, noise or music it emits the subtitle
credits it saw most: «ترجمة نانسي قنقر» ("Translation by Nancy Qunqar") above all. VAD removes
most of these at the source; this filter is the second line for the ones that slip through on
real voice notes (a credit appended after the last word, a "subscribe" line over background
music).

Only *credit-style* phrases are listed: text nobody says in a voice note to an assistant.
Short generic phrases the models also produce on noise («شكرا لك», «شنوة؟») are deliberately
NOT listed — a person really says those, and VAD already handles the noise case.
"""

from __future__ import annotations

from .arabic import clean_spacing, fold_token

# Folded forms (see arabic.fold). Keep entries specific enough that real speech never matches.
HALLUCINATION_PHRASES: tuple[str, ...] = (
    "ترجمه نانسي قنقر",
    "ترجمه نانسي قنقار",
    "نانسي قنقر",
    "اشتركوا في القناه",
    "اشترك في القناه",
    "لا تنسوا الاشتراك في القناه",
    "لا تنسى الاشتراك في القناه",
    "شكرا للمشاهده",
    "شكرا على المشاهده",
    "شكرا لكم على المشاهده",
    "شكرا لمشاهدتكم",
    "تمت الترجمه بواسطه",
    "موسيقى",
    "موسيقي",
)
_PHRASES = sorted((tuple(p.split()) for p in HALLUCINATION_PHRASES), key=len, reverse=True)


def strip_hallucinations(text: str) -> tuple[str, list[str]]:
    """Return ``(clean_text, removed)``. Credits are removed only when they are the whole transcript
    or sit at its very start or end, where Whisper puts them; the same words mid-speech stay."""
    tokens = (text or "").split()
    removed: list[str] = []
    changed = True
    while changed and tokens:
        changed = False
        folded = [fold_token(t) for t in tokens]
        # Ignore pure-punctuation tokens at the edges so «. ترجمة نانسي قنقر» still matches.
        lo, hi = 0, len(tokens)
        while lo < hi and not folded[lo]:
            lo += 1
        while hi > lo and not folded[hi - 1]:
            hi -= 1
        core = folded[lo:hi]
        for p in _PHRASES:
            n = len(p)
            if n <= len(core) and tuple(core[-n:]) == p:
                removed.append(" ".join(tokens[hi - n:hi]))
                tokens = tokens[:hi - n]
                changed = True
                break
            if n <= len(core) and tuple(core[:n]) == p:
                removed.append(" ".join(tokens[lo:lo + n]))
                tokens = tokens[lo + n:]
                changed = True
                break
    out = clean_spacing(" ".join(tokens))
    if out and not any(fold_token(t) for t in out.split()):
        out = ""  # only punctuation left
    return out, removed
