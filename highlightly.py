import requests

BASE_URL = "https://nhl-ncaah-api.p.rapidapi.com"


class HighlightlyClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "nhl-ncaah-api.p.rapidapi.com",
        }

    def _get(self, endpoint, params=None):
        response = requests.get(f"{BASE_URL}{endpoint}", headers=self.headers, params=params or {})
        response.raise_for_status()
        return response.json()

    def get_teams(self, league="NHL"):
        return self._get("/teams", params={"league": league})

    def get_standings(self, year, league_type="NHL", limit=40):
        return self._get("/standings", params={
            "leagueType": league_type, "year": year, "offset": 0, "limit": limit
        })

    def get_matches(self, league="NHL", season=None, date_str=None, limit=100, offset=0,
                     home_team_id=None, away_team_id=None):
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

    def get_match_by_id(self, match_id):
        result = self._get(f"/matches/{match_id}")
        return result[0] if isinstance(result, list) else result

    def get_head_to_head(self, home_team_id, away_team_id, league="NHL", limit=20):
        return self.get_matches(league=league, home_team_id=home_team_id,
                                 away_team_id=away_team_id, limit=limit)

    def get_team_recent_matches(self, team_id, league="NHL", season=None, n=10):
        home = self.get_matches(league=league, season=season, home_team_id=team_id, limit=50)
        away = self.get_matches(league=league, season=season, away_team_id=team_id, limit=50)
        all_matches = home.get("data", []) + away.get("data", [])
        finished = [m for m in all_matches if m.get("state", {}).get("report") == "Final"]
        finished.sort(key=lambda m: m["date"], reverse=True)
        return finished[:n]

    def get_all_team_stats(self, year, league_type="NHL"):
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
              
