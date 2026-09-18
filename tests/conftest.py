"""Shared fixtures."""

import pytest


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that resolve spoken dates ("tomorrow at 5", "next Thursday") against a
    lecture's start time expect fixed calendar dates; without this they fail as the
    real date moves on. Only the store's clock is frozen, at the day they were written."""
    monkeypatch.setattr("lecture_copilot.store.now_iso", lambda: "2026-09-16T10:00:00+00:00")
