import pytest


@pytest.fixture
def filters(lahja):
    import hermes_plugins.lahja.filters as f
    return f


@pytest.fixture
def corr(lahja):
    import hermes_plugins.lahja.corrections as c
    return c


# -- hallucination filter ---------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "ترجمة نانسي قنقر",
    " ترجمة نانسي قنقر ",
    "ترجمة نانسي قنقر.",
    "اشتركوا في القناة",
    "شكراً للمشاهدة!",
    "[موسيقى]",
])
def test_whole_transcript_hallucination_is_dropped(filters, text):
    out, removed = filters.strip_hallucinations(text)
    assert out == ""
    assert removed


def test_trailing_credit_is_stripped_but_speech_kept(filters):
    out, removed = filters.strip_hallucinations("بدي روح على المول بكرا ترجمة نانسي قنقر")
    assert out == "بدي روح على المول بكرا"
    assert removed == ["ترجمة نانسي قنقر"]


def test_leading_and_trailing_credits(filters):
    out, _ = filters.strip_hallucinations("ترجمة نانسي قنقر وين صرت؟ تمت الترجمة بواسطة")
    assert out == "وين صرت؟"


def test_mid_sentence_words_are_never_touched(filters):
    text = "قالتلي شكرا للمشاهدة يا ريم بس الفيلم كان طويل"
    assert filters.strip_hallucinations(text) == (text, [])


def test_real_short_phrases_are_kept(filters):
    for text in ("شكرا لك.", "شنوة؟", "تمام"):
        assert filters.strip_hallucinations(text)[0] == text


# -- corrections ------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,wrong,right", [
    ("جي إلي تي => JLT", "جي إلي تي", "JLT"),
    ("GLT -> JLT", "GLT", "JLT"),
    ("«دبي مرينا» → Dubai Marina", "دبي مرينا", "Dubai Marina"),
    ("JLT <- جي إل تي", "جي إل تي", "JLT"),
])
def test_parse_rule(corr, raw, wrong, right):
    c = corr.parse_rule(raw)
    assert (c.wrong, c.right) == (wrong, right)


@pytest.mark.parametrize("raw", ["", "JLT", "=> JLT", "x =>", "، => JLT", "a" * 200 + " => b"])
def test_parse_rule_rejects(corr, raw):
    assert corr.parse_rule(raw) is None


def test_apply_replaces_respaced_variants(corr):
    rules = [corr.Correction("جي إلي تي", "JLT")]
    for heard in ("بدي روح على جي إلي تي بكرا", "بدي روح على جي إليتي بكرا",
                  "بدي روح على جيإلي تي بكرا", "بدي روح على جيإليتي بكرا"):
        out, n = corr.apply(heard, rules)
        assert n == 1, heard
        assert out == "بدي روح على JLT بكرا"
    # A different spelling (missing ي) is a different phrase: teach it separately, no fuzzy guessing.
    assert corr.apply("على جي إلتي", rules) == ("على جي إلتي", 0)


def test_apply_folds_hamza_and_keeps_punctuation(corr):
    rules = [corr.Correction("جي إلي تي", "JLT")]
    out, n = corr.apply("رحت على جي الي تي، وبعدين البيت.", rules)
    assert (out, n) == ("رحت على JLT، وبعدين البيت.", 1)


def test_apply_never_matches_inside_a_word(corr):
    rules = [corr.Correction("تي", "T")]
    assert corr.apply("بدي روح على البيتي", rules) == ("بدي روح على البيتي", 0)


def test_apply_latin_is_case_insensitive(corr):
    rules = [corr.Correction("GLT", "JLT")]
    assert corr.apply("meet me at glt tower", rules) == ("meet me at JLT tower", 1)


def test_longest_rule_wins(corr):
    rules = [corr.Correction("دبي", "Dubai"), corr.Correction("دبي مرينا", "Dubai Marina")]
    assert corr.apply("على دبي مرينا", rules)[0] == "على Dubai Marina"


def test_state_roundtrip_and_dedupe(corr, ctx):
    corr.add(ctx.state, corr.Correction("جي إلي تي", "JLT"))
    corr.add(ctx.state, corr.Correction("جي الي تي", "JLT Tower"))  # same folded key → replaces
    items = corr.load(ctx.state)
    assert [(c.wrong, c.right) for c in items] == [("جي الي تي", "JLT Tower")]
    assert corr.remove(ctx.state, 1).right == "JLT Tower"
    assert corr.load(ctx.state) == []
    assert corr.remove(ctx.state, 5) is None


def test_load_ignores_garbage_state(corr, ctx):
    ctx.state.set("corrections", [{"wrong": 1}, "x", {"wrong": "a", "right": "b"}, {"wrong": "،", "right": "b"}])
    assert [(c.wrong, c.right) for c in corr.load(ctx.state)] == [("a", "b")]
