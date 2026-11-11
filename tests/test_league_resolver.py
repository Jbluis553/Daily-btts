from __future__ import annotations

from unittest.mock import MagicMock

from src.top_leagues import LEAGUES, LeagueResolver


def test_resolver_uses_api_ids_not_hardcoded():
    """Los ids vienen de la API, nunca de constantes en el código."""
    client = MagicMock()
    client.get.return_value = {
        "data": [
            {"id": "real_id_123", "name": "Premier League"},
            {"id": "real_id_456", "name": "La Liga"},
        ]
    }
    resolver = LeagueResolver(client)
    matches = [
        {"id": "m1", "competition_id": "real_id_123"},
        {"id": "m2", "competition_id": "fake_hardcoded_id"},
        {"id": "m3", "competition_id": "real_id_456"},
    ]
    out = resolver.filter_matches(matches)
    assert [m["id"] for m in out] == ["m1", "m3"]
    assert LEAGUES["premier_league"]["competition_id"] == "real_id_123"


def test_resolver_falls_back_to_all_when_api_fails():
    client = MagicMock()
    client.get.side_effect = Exception("network down")
    resolver = LeagueResolver(client)
    matches = [{"id": "m1", "competition_id": "x"}]
    assert resolver.filter_matches(matches) == matches
