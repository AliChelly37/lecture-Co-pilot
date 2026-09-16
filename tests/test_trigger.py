from lecture_copilot.trigger import score_segment

HITS = [
    "Problem set four is due next Thursday at five p.m., submitted through the course website.",
    "The midterm exam is on October fourteenth.",
    "Actually, let me correct that, the midterm has moved to October twenty first.",
    "The reading for next time is chapter eleven, sections one through four.",
    "Don't forget, the lab report deadline is end of week seven.",
    # Hypotheticals are still hits: the badge is a hint, the model judges intent later.
    "If this problem set were due tomorrow, you would all be panicking.",
]

MISSES = [
    "Notice that this integral vanishes for an ideal gas because Z is identically one.",
    "Let me now go to the board and write the residual Gibbs energy expression.",
    "Okay, let us work through an example with nitrogen at three hundred kelvin.",
    "The Navier-Stokes equations are not relevant here.",
]


def test_hits() -> None:
    for text in HITS:
        assert score_segment(text).hit, text


def test_misses() -> None:
    for text in MISSES:
        assert not score_segment(text).hit, text


def test_terms_reported() -> None:
    r = score_segment("Problem set four is due next Thursday.")
    assert "due" in r.terms and "problem set" in r.terms
    assert r.has_date
