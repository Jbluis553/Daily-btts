from __future__ import annotations

from unittest.mock import MagicMock

from src.data_fetcher import DataFetcher


def test_data_fetcher_accepts_date_range_contract() -> None:
    client = MagicMock()
    client.get.return_value = {"data": [], "meta": {"total_pages": 1}}

    fetcher = DataFetcher(client)
    result = fetcher.get_upcoming_matches(date_from="2026-10-09", date_to="2026-10-10")

    assert result == []
    client.get.assert_called_once()
    assert client.get.call_args.kwargs["params"]["date_from"] == "2026-10-09"
    assert client.get.call_args.kwargs["params"]["date_to"] == "2026-10-10"


def test_team_history_marks_owner_team_id() -> None:
    client = MagicMock()
    client.get.return_value = {
        "data": [
            {"id": "m1", "home_team": {"id": "9"}, "away_team": {"id": "8"},
             "score": {"home": 1, "away": 2}, "utc_date": "2026-10-01"},
            {"id": "m0", "home_team": {"id": "8"}, "away_team": {"id": "9"},
             "score": {"home": 0, "away": 0}, "utc_date": "2026-09-28"},
        ]
    }
    fetcher = DataFetcher(client)
    hist = fetcher.get_team_finished_matches("8", n=10)
    assert hist[0]["_team_id"] == "8"
