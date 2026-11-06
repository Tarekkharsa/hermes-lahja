"""Detect English speech that the dialect model wrote out in Arabic letters.

Whisper decides the language from the audio, and English spoken with an Arabic accent is often
detected as Arabic (on a real voice note: dialect model p=0.9999, stock large-v3 p=0.82). The
dialect model then writes the English phonetically: «ذاتس لايك يو غايز… آي دونت نو». That text is
easy to recognise: it is dense with English function words spelled in Arabic letters.

Only spellings that are NOT ordinary Arabic words are listed. Deliberately excluded because they
are real Arabic: إي (yes), شي (thing), تو (just now), كان (was), ذات (self), وانت (and you),
فور (immediately), اي (which/yes), ماي (water, Gulf), سو, يا, لايك (a "like", common slang),
بات (became), سيد (mister), كانت (she was), وانا (and I).
"""

from __future__ import annotations

from .arabic import fold_token

# Folded forms (see arabic.fold): أإآ→ا, ة→ه, ى→ي, no diacritics.
MARKERS: frozenset[str] = frozenset({
    "ذا", "ذي", "ذه",                  # the
    "اند",                              # and
    "يو", "يور",                        # you, your
    "ذاتس", "اتس", "ايتس",              # that's, it's
    "ذس", "ذيس",                        # this
    "ذن", "ثن",                         # then
    "وي",                               # we
    "از", "ايز", "واز",                  # is, was
    "دونت", "وونت", "ديدنت",              # don't, won't, didn't
    "نو",                               # know / no
    "فيري", "فري",                       # very
    "وات", "وت", "وير", "وين",           # what, where (وين excluded below: Levantine "where")
    "ويذ", "وذ",                        # with
    "هاف", "هاڤ", "هاز",                 # have, has
    "جست", "اباوت", "بيكوز", "بيكاز",     # just, about, because
    "ذاي", "ذير", "ذيم", "ذين",          # they, there, them, then
    "انثر", "اناذر", "اذر",              # another, other
    "غونا", "جونا",                      # gonna
    "ثينك", "سينك", "نيد",              # think, need
    "ايم",                              # I'm
    "ايف",                              # if
}) - {"وين"}  # «وين» is everyday Levantine for "where": never a marker

MIN_TOKENS = 5
MIN_DISTINCT = 3
MIN_RATIO = 0.2


def english_score(text: str) -> tuple[float, int]:
    """``(share of tokens that are transliterated English markers, distinct markers seen)``."""
    tokens = [t for t in (fold_token(x) for x in (text or "").split()) if t]
    if not tokens:
        return 0.0, 0
    hits = [t for t in tokens if t in MARKERS]
    return len(hits) / len(tokens), len(set(hits))


def looks_like_transliterated_english(text: str) -> bool:
    tokens = [t for t in (text or "").split() if fold_token(t)]
    if len(tokens) < MIN_TOKENS:
        return False
    ratio, distinct = english_score(text)
    return ratio >= MIN_RATIO and distinct >= MIN_DISTINCT
