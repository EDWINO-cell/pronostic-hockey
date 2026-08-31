"""
Connecteur pour l'API Highlightly (NHL & NCAAH API via RapidAPI).
Endpoints validés manuellement le 24/08/2026 sur Colab.

Documentation des quirks découverts :
- /teams          : query params -> league
- /standings      : query params -> leagueType, year, offset, limit
                     (PAS "league"/"season")
- /matches        : query params -> league, season, date, offset, limit
                     date seul (sans season) ne renvoie rien
- /matches/{id}   : PATH param -> contient overallStatistics, events,
                     predictions, venue, forecast (pas d'endpoint /statistics séparé)
- /lineups/{id}   : PATH param -> compositions dispo ~30min avant le match
                     seulement, vides sinon
- /players        : query params -> name, offset, limit (PAS "league")
- /players/{id}   : PATH param -> profil complet (équipe, draft, position, etc.)
"""
import os
import requests
from datetime import date, timedelta

BASE_URL = "https://nhl-ncaah-api.p.rapidapi.com"


class HighlightlyClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("RAPIDAPI_KEY")
        if not self.api_key:
            raise ValueError("Clé API manquante : passe api_key ou définis RAPIDAPI_KEY")
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "nhl-ncaah-api.p.rapidapi.com",
        }

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params or {})
        response.raise_for_status()
        return response.json()

    # --- Équipes ---
    def get_teams(self, league: str = "NHL") -> list[dict]:
        return self._get("/teams", params={"league": league})

    # --- Classements ---
    def get_standings(self, year: int, league_type: str = "NHL",
                       league_name: str | None = None, abbreviation: str | None = None,
                       limit: int = 40) -> dict:
        params = {"leagueType": league_type, "year": year, "offset": 0, "limit": limit}
        if league_name:
            params["leagueName"] = league_name
        if abbreviation:
            params["abbreviation"] = abbreviation
        return self._get("/standings", params=params)

    # --- Matchs ---
    def get_matches(self, league: str = "NHL", season: int | None = None,
                     date_str: str | None = None, limit: int = 100, offset: int = 0,
                     home_team_id: int | None = None, away_team_id: int | None = None) -> dict:
        params = {"league": league, "limit": limit, "offset": offset}
        if season:
            params["season"] = season
        if date_str:
            params["date"] = date_str
        if home_team_id:
            params["homeTeamId"] = home_team_id
        if away_team_id:
            params["awayTeamId"] = away_team_id
        return self._get("/matches", params=params)

    def get_match_by_id(self, match_id: int) -> dict:
        """Contient overallStatistics, events (play-by-play), predictions, venue, forecast."""
        result = self._get(f"/matches/{match_id}")
        return result[0] if isinstance(result, list) else result

    def get_head_to_head(self, home_team_id: int, away_team_id: int,
                          league: str = "NHL", limit: int = 20) -> dict:
        """Historique des confrontations directes entre deux équipes."""
        return self.get_matches(league=league, home_team_id=home_team_id,
                                  away_team_id=away_team_id, limit=limit)

    # --- Compositions ---
    def get_lineups(self, match_id: int) -> dict:
        """Compositions disponibles seulement ~30 min avant le match."""
        return self._get(f"/lineups/{match_id}")

    # --- Joueurs ---
    def search_players(self, name: str, limit: int = 10) -> dict:
        return self._get("/players", params={"name": name, "limit": limit})

    def get_player_by_id(self, player_id: int) -> dict:
        result = self._get(f"/players/{player_id}")
        return result[0] if isinstance(result, list) else result

    # --- Utilitaire : derniers matchs d'une équipe (pour la forme récente) ---
    def get_team_recent_matches(self, team_id: int, league: str = "NHL",
                                  season: int | None = None, n: int = 10) -> list[dict]:
        """Récupère les derniers matchs (terminés) d'une équipe, home ou away confondus."""
        home = self.get_matches(league=league, season=season, home_team_id=team_id, limit=50)
        away = self.get_matches(league=league, season=season, away_team_id=team_id, limit=50)
        all_matches = home.get("data", []) + away.get("data", [])
        finished = [m for m in all_matches if m.get("state", {}).get("report") == "Final"]
        finished.sort(key=lambda m: m["date"], reverse=True)
        return finished[:n]

    # --- Utilitaire : stats agrégées par équipe, pour le modèle Poisson ---
    def get_all_team_stats(self, year: int, league_type: str = "NHL") -> dict[int, dict]:
        """Extrait goals_for / goals_against / games_played pour CHAQUE équipe
        à partir de /standings (toutes conférences confondues).
        Retourne {team_id: {"goals_for": int, "goals_against": int, "games_played": int}}"""
        standings = self.get_standings(year=year, league_type=league_type, limit=40)
        team_stats = {}
        for conference in standings.get("data", []):
            for entry in conference.get("data", []):
                team_id = entry["team"]["id"]
                stats_by_name = {s["displayName"]: s["value"] for s in entry["statistics"]}
                team_stats[team_id] = {
                    "goals_for": int(stats_by_name.get("Goals For", 0)),
                    "goals_against": int(stats_by_name.get("Goals Against", 0)),
                    "games_played": int(stats_by_name.get("Games Played", 0)),
                }
        return team_stats
