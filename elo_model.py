"""
Système Elo (avec marge de victoire) pour prédire l'issue d'un match NHL.

Calibré par validation croisée (split 70/30 sur la saison 2025-2026) :
K=10, home_advantage=25 points Elo. Meilleur modèle d'issue de match produit
jusqu'ici (log-loss 0.678 sur l'échantillon de validation, contre 0.857 pour
le modèle Poisson+xG).

N'a besoin que des SCORES finaux (pas du play-by-play complet) — beaucoup
plus rapide à calculer en production que le xG.
"""
import math
import time
import requests
from xg_model import NHL_TEAM_ABBREVS, HIGHLIGHTLY_TO_NHL_ABBREV_OVERRIDES, NHL_REQUEST_PAUSE

INITIAL_ELO = 1500
K_FACTOR = 10
HOME_ADVANTAGE_ELO = 25


def elo_expected_home_win_prob(home_elo, away_elo, home_advantage=HOME_ADVANTAGE_ELO):
    adjusted_home_elo = home_elo + home_advantage
    return 1 / (1 + 10 ** ((away_elo - adjusted_home_elo) / 400))


def elo_update(home_elo, away_elo, home_won, goal_diff, k=K_FACTOR, home_advantage=HOME_ADVANTAGE_ELO):
    expected_home = elo_expected_home_win_prob(home_elo, away_elo, home_advantage)
    actual_home = 1.0 if home_won else 0.0
    mov_multiplier = math.log(goal_diff + 1)
    k_effective = k * mov_multiplier
    new_home_elo = home_elo + k_effective * (actual_home - expected_home)
    new_away_elo = away_elo + k_effective * (expected_home - actual_home)
    return new_home_elo, new_away_elo


def get_live_team_elo_ratings(season_id):
    """Calcule les cotes Elo actuelles de chaque équipe en rejouant tous les
    matchs terminés de la saison en cours, dans l'ordre chronologique."""
    all_games = []
    seen_game_ids = set()

    for abbrev in NHL_TEAM_ABBREVS:
        resp = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/{abbrev}/{season_id}")
        resp.raise_for_status()
        games = resp.json().get("games", [])
        for g in games:
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                continue
            if g["id"] in seen_game_ids:
                continue
            seen_game_ids.add(g["id"])
            home_score = g.get("homeTeam", {}).get("score")
            away_score = g.get("awayTeam", {}).get("score")
            if home_score is None or away_score is None:
                continue
            all_games.append({
                "date": g["startTimeUTC"],
                "home_abbrev": g["homeTeam"]["abbrev"],
                "away_abbrev": g["awayTeam"]["abbrev"],
                "home_score": home_score,
                "away_score": away_score,
            })
        time.sleep(NHL_REQUEST_PAUSE)

    all_games.sort(key=lambda g: g["date"])

    elo_ratings = {ab: INITIAL_ELO for ab in NHL_TEAM_ABBREVS}
    for g in all_games:
        home_ab, away_ab = g["home_abbrev"], g["away_abbrev"]
        if home_ab not in elo_ratings or away_ab not in elo_ratings:
            continue
        home_won = g["home_score"] > g["away_score"]
        goal_diff = abs(g["home_score"] - g["away_score"])
        new_home, new_away = elo_update(elo_ratings[home_ab], elo_ratings[away_ab], home_won, goal_diff)
        elo_ratings[home_ab] = new_home
        elo_ratings[away_ab] = new_away

    return elo_ratings


def remap_elo_to_highlightly_ids(elo_by_abbrev, highlightly_teams):
    result = {}
    for team in highlightly_teams:
        hl_abbrev = team.get("abbreviation")
        nhl_abbrev = HIGHLIGHTLY_TO_NHL_ABBREV_OVERRIDES.get(hl_abbrev, hl_abbrev)
        if nhl_abbrev in elo_by_abbrev:
            result[team["id"]] = elo_by_abbrev[nhl_abbrev]
    return result
