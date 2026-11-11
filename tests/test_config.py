from __future__ import annotations

from datetime import datetime, timezone

from src.config import Settings


def test_settings_defaults_use_today() -> None:
    settings = Settings.from_dict({})
    assert settings.date_from == datetime.now(timezone.utc).date().isoformat()
    assert settings.date_to == settings.date_from
    assert settings.min_pct == 80.0
    assert settings.top_n == 20
    assert settings.log_level == "INFO"


def test_settings_explicit_values_are_preserved() -> None:
    settings = Settings.from_dict(
        {
            "date_from": "2026-10-10",
            "date_to": "2026-10-12",
            "min_pct": "90",
            "top_n": "5",
            "log_level": "debug",
        }
    )
    assert settings.date_from == "2026-10-10"
    assert settings.date_to == "2026-10-12"
    assert settings.min_pct == 90.0
    assert settings.top_n == 5
    assert settings.log_level == "DEBUG"


def test_settings_rejects_inverted_range() -> None:
    try:
        Settings.from_dict({"date_from": "2026-10-12", "date_to": "2026-10-10"})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("debía rechazar date_from > date_to")


def test_settings_rejects_bad_min_pct() -> None:
    for bad in ("0", "101", "-5"):
        try:
            Settings.from_dict({"min_pct": bad})
        except ValueError:
            continue
        raise AssertionError(f"debía rechazar min_pct={bad}")
