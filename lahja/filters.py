"""Known Whisper hallucinations in Arabic.

Whisper was trained on subtitled video, so on silence, noise or music it emits the subtitle
credits it saw most: «ترجمة نانسي قنقر» ("Translation by Nancy Qunqar") above all. VAD removes
most of these at the source; this filter is the second line for the ones that slip through.

Two lists, because some of these phrases are also things people really say:

* ``EDGE_PHRASES`` are subtitle credits nobody says to an assistant. They are removed when they
  are the whole transcript or sit at its very start or end, where Whisper appends them.
* ``WHOLE_ONLY_PHRASES`` («موسيقى», "subscribe to the channel", "thanks for watching") can be a
  real request («شغل موسيقى» = "play music»; «كيف اشترك في القناة» = "how do I subscribe to the
  channel"). They are removed only when they are the *entire* transcript.

Short generic phrases the models also produce on noise («شكرا لك», «شنوة؟») are not listed at
all — a person really says those, and VAD already handles the noise case.
"""

from __future__ import annotations

from .arabic import clean_spacing, fold_token

# Folded forms (see arabic.fold).
EDGE_PHRASES: tuple[str, ...] = (
    "ترجمه نانسي قنقر",
    "ترجمه نانسي قنقار",
    "نانسي قنقر",
    "تمت الترجمه بواسطه",
)
WHOLE_ONLY_PHRASES: tuple[str, ...] = (
    "اشتركوا في القناه",
    "اشترك في القناه",
    "لا تنسوا الاشتراك في القناه",
    "لا تنسى الاشتراك في القناه",
    "شكرا للمشاهده",
    "شكرا على المشاهده",
    "شكرا لكم على المشاهده",
    "شكرا لمشاهدتكم",
    "موسيقى",
    "موسيقي",
)
HALLUCINATION_PHRASES = EDGE_PHRASES + WHOLE_ONLY_PHRASES  # kept for callers/tests
_EDGE = sorted((tuple(p.split()) for p in EDGE_PHRASES), key=len, reverse=True)
_WHOLE = {tuple(p.split()) for p in WHOLE_ONLY_PHRASES}


def _core(tokens: list[str]) -> tuple[int, int, list[str]]:
    """Index range of the tokens that carry letters (edge punctuation ignored) and their folds."""
    folded = [fold_token(t) for t in tokens]
    lo, hi = 0, len(tokens)
    while lo < hi and not folded[lo]:
        lo += 1
    while hi > lo and not folded[hi - 1]:
        hi -= 1
    return lo, hi, folded[lo:hi]


def strip_hallucinations(text: str) -> tuple[str, list[str]]:
    """Return ``(clean_text, removed)``. Never touches words in the middle of speech."""
    tokens = (text or "").split()
    removed: list[str] = []
    lo, hi, core = _core(tokens)
    letters = tuple(t for t in core if t)
    if letters in _WHOLE:
        return "", [" ".join(tokens)]
    changed = True
    while changed and tokens:
        changed = False
        lo, hi, core = _core(tokens)
        for p in _EDGE:
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
    elif tuple(t for t in (fold_token(x) for x in out.split()) if t) in _WHOLE and removed:
        out, removed = "", removed + [out]  # e.g. «موسيقى ترجمة نانسي قنقر»: nothing real is left
    return out, removed
