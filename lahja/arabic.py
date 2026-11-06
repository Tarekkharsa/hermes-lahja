"""Arabic text normalization shared by the hallucination filter and the corrections matcher.

Matching is done on a *folded* form (no diacritics/tatweel, alef/ya/ta-marbuta variants unified,
punctuation dropped) so a correction taught as «جي إل تي» also matches «جي إلي تي»'s neighbours
written with different hamza seats or punctuation. The transcript itself is never folded: only
the matched span is replaced, everything else keeps the model's own spelling.
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS_RE = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in (
    (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED), (0x0640, 0x0640),
)) + "]")  # harakat, tanwin, shadda, dagger alef, Quranic marks, tatweel
_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ؤ": "و", "ئ": "ي", "ة": "ه",
})
_ZERO_WIDTH_RE = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in (
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2066, 0x2069), (0xFEFF, 0xFEFF),
)) + "]")  # zero-width and bidi controls that ride along in Arabic text
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def fold(text: str) -> str:
    """Lower-cased, diacritic-free, letter-folded, punctuation-free form used only for matching."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _DIACRITICS_RE.sub("", text).translate(_FOLD).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def fold_token(token: str) -> str:
    return fold(token).replace(" ", "")


def clean_spacing(text: str) -> str:
    text = _WS_RE.sub(" ", text or "").strip()
    return re.sub(r"\s+([،؛؟.,!?:])", r"\1", text)
