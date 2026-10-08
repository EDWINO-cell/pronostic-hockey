"""
Modèle xG en Python pur (régression logistique réimplémentée à partir des
coefficients exportés de scikit-learn) — évite la dépendance scikit-learn
dans l'app Streamlit, plus léger à déployer.

Entraîné sur 153 754 tirs de la saison NHL 2025-2026 (API officielle NHL).
Validé par backtest sans lookahead : améliore la précision du modèle de
prédiction de 53.3% à 54.8% par rapport aux buts réels bruts.
"""
import math

XG_INTERCEPT = -0.705729460510384

XG_COEFFICIENTS = {
    "shot_type_backhand": 0.39303425106428364,
    "shot_type_bat": 0.19095559214930366,
    "shot_type_between-legs": 0.038995772145797115,
    "shot_type_cradle": 0.016283352085678406,
    "shot_type_deflected": 0.3782216900701241,
    "shot_type_poke": 0.23565237787066176,
    "shot_type_slap": 0.9398511739705263,
    "shot_type_snap": 0.9625434259325092,
    "shot_type_tip-in": -0.2050059702080281,
    "shot_type_unknown": -4.170127338486452,
    "shot_type_wrap-around": -0.18977715346863053,
    "shot_type_wrist": 0.72282383855579,
    "situation_even": -0.788542940768578,
    "situation_power_play": -0.4674086025227615,
    "situation_short_handed": 0.5694025549732483,
    "distance": -0.048021290195098924,
    "angle": -0.016019830430102088,
    "is_rebound": -0.14556403481453095,
}


def predict_xg(distance: float, angle: float, shot_type: str, situation: str, is_rebound: bool) -> float:
    """Probabilité de but pour un tir donné (0 à 1)."""
    z = XG_INTERCEPT
    z += XG_COEFFICIENTS.get(f"shot_type_{shot_type}", 0.0)
    z += XG_COEFFICIENTS.get(f"situation_{situation}", 0.0)
    z += XG_COEFFICIENTS["distance"] * distance
    z += XG_COEFFICIENTS["angle"] * angle
    if is_rebound:
        z += XG_COEFFICIENTS["is_rebound"]
    return 1 / (1 + math.exp(-z))


def shot_distance_angle(x: float, y: float) -> tuple[float, float]:
    goal_x = 89 if x >= 0 else -89
    distance = math.sqrt((goal_x - x) ** 2 + y ** 2)
    angle = math.degrees(math.atan2(abs(y), abs(goal_x - x))) if (goal_x - x) != 0 or y != 0 else 0
    return round(distance, 1), round(angle, 1)


def parse_situation_code(code: str, event_team_is_home: bool) -> str:
    away_goalie, away_skaters, home_skaters, home_goalie = (int(c) for c in code)
    if away_skaters == home_skaters:
        return "even"
    shooting_team_skaters = home_skaters if event_team_is_home else away_skaters
    opponent_skaters = away_skaters if event_team_is_home else home_skaters
    return "power_play" if shooting_team_skaters > opponent_skaters else "short_handed"


SHOT_TYPES = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}


def extract_match_xg(play_by_play: dict) -> dict:
    """Calcule le xG total pour chaque équipe à partir du play-by-play d'un match
    (sortie de /v1/gamecenter/{gameId}/play-by-play)."""
    home_id = play_by_play["homeTeam"]["id"]
    away_id = play_by_play["awayTeam"]["id"]
    plays = play_by_play.get("plays", [])

    last_shot_time_by_team = {}
    xg_by_team = {home_id: 0.0, away_id: 0.0}
    goals_by_team = {home_id: 0, away_id: 0}

    for p in plays:
        if p.get("typeDescKey") not in SHOT_TYPES:
            continue
        d = p.get("details", {})
        x, y = d.get("xCoord"), d.get("yCoord")
        if x is None or y is None:
            continue

        team_id = d.get("eventOwnerTeamId")
        if team_id not in xg_by_team:
            continue
        is_home = team_id == home_id
        distance, angle = shot_distance_angle(x, y)
        situation = parse_situation_code(p["situationCode"], is_home)
        shot_type = d.get("shotType", "unknown")

        mm, ss = (int(v) for v in p["timeInPeriod"].split(":"))
        total_seconds = p["periodDescriptor"]["number"] * 1200 + mm * 60 + ss
        is_rebound = (team_id in last_shot_time_by_team
                      and total_seconds - last_shot_time_by_team[team_id] <= 3)
        last_shot_time_by_team[team_id] = total_seconds

        xg_by_team[team_id] += predict_xg(distance, angle, shot_type, situation, is_rebound)
        if p.get("typeDescKey") == "goal":
            goals_by_team[team_id] += 1

    return {
        "home_team_id": home_id, "away_team_id": away_id,
        "home_xg": round(xg_by_team[home_id], 3), "away_xg": round(xg_by_team[away_id], 3),
        "home_goals": goals_by_team[home_id], "away_goals": goals_by_team[away_id],
    }


# --- Collecte en direct (saison en cours) + mapping Highlightly <-> NHL officiel ---
import requests

NHL_TEAM_ABBREVS = ["ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA",
                     "LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS",
                     "STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH"]

# Highlightly utilise des abréviations différentes de NHL officiel dans de rares cas
HIGHLIGHTLY_TO_NHL_ABBREV_OVERRIDES = {"UTAH": "UTA", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "LA": "LAK"}


def get_live_team_xg_stats(season_id: str) -> dict:
    """Parcourt le calendrier de chaque équipe, récupère le xG de chaque match
    terminé de la saison en cours, agrège par équipe (clé = abréviation, pas
    l'id NHL, pour faciliter le mapping avec Highlightly côté app)."""
    stats_by_abbrev = {ab: {"goals_for": 0.0, "goals_against": 0.0, "games_played": 0}
                        for ab in NHL_TEAM_ABBREVS}
    seen_game_ids = set()
    abbrev_by_nhl_id = {}

    for abbrev in NHL_TEAM_ABBREVS:
        resp = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/{abbrev}/{season_id}")
        resp.raise_for_status()
        games = resp.json().get("games", [])
        for g in games:
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                continue
            abbrev_by_nhl_id[g["homeTeam"]["id"]] = g["homeTeam"]["abbrev"]
            abbrev_by_nhl_id[g["awayTeam"]["id"]] = g["awayTeam"]["abbrev"]
            if g["id"] in seen_game_ids:
                continue
            seen_game_ids.add(g["id"])

    for game_id in seen_game_ids:
        try:
            resp = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
            resp.raise_for_status()
            pbp = resp.json()
            match_xg = extract_match_xg(pbp)
        except Exception:
            continue

        home_ab = abbrev_by_nhl_id.get(match_xg["home_team_id"])
        away_ab = abbrev_by_nhl_id.get(match_xg["away_team_id"])
        if home_ab in stats_by_abbrev:
            stats_by_abbrev[home_ab]["goals_for"] += match_xg["home_xg"]
            stats_by_abbrev[home_ab]["goals_against"] += match_xg["away_xg"]
            stats_by_abbrev[home_ab]["games_played"] += 1
        if away_ab in stats_by_abbrev:
            stats_by_abbrev[away_ab]["goals_for"] += match_xg["away_xg"]
            stats_by_abbrev[away_ab]["goals_against"] += match_xg["home_xg"]
            stats_by_abbrev[away_ab]["games_played"] += 1

    return stats_by_abbrev


def remap_stats_to_highlightly_ids(stats_by_abbrev: dict, highlightly_teams: list) -> dict:
    """Convertit les stats xG (clé = abréviation NHL) vers des clés = id Highlightly,
    pour pouvoir les utiliser directement dans compute_team_strengths() à la place
    des stats Highlightly (goals_for/against bruts)."""
    result = {}
    for team in highlightly_teams:
        hl_abbrev = team.get("abbreviation")
        nhl_abbrev = HIGHLIGHTLY_TO_NHL_ABBREV_OVERRIDES.get(hl_abbrev, hl_abbrev)
        if nhl_abbrev in stats_by_abbrev and stats_by_abbrev[nhl_abbrev]["games_played"] > 0:
            result[team["id"]] = stats_by_abbrev[nhl_abbrev]
    return result
