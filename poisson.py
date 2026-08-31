"""
Modèle de prédiction Poisson pour le hockey sur glace.

Principe : le nombre de buts marqués par une équipe suit approximativement
une loi de Poisson. On estime le nombre de buts ATTENDUS (lambda) pour
chaque équipe à partir de :
  1. La force d'attaque/défense de chaque équipe sur la saison (calibrage)
  2. Un ajustement à partir des features du match (forme récente, repos,
     face-à-face) calculées dans model/features.py

Une fois les deux lambdas obtenus, on calcule la matrice de probabilité
de tous les scores possibles (0-0, 1-0, 0-1, 2-1, ...), puis on en déduit :
  - probabilités victoire domicile / nul / victoire extérieur
  - probabilité over/under sur le total de buts
  - score exact le plus probable

Limites assumées : ce modèle ne capture pas les événements rares (buteur
précis, buts en power play isolés) ni l'imprévisible (blessure de dernière
minute). Précision réaliste attendue sur l'issue du match : ~55-62%.
"""
import math
from scipy.stats import poisson


def compute_league_average(all_team_stats: dict[int, dict]) -> float:
    """Moyenne de buts marqués par équipe et par match, sur toute la ligue.
    all_team_stats : {team_id: {"goals_for": int, "games_played": int, ...}}"""
    total_goals = sum(t["goals_for"] for t in all_team_stats.values())
    total_games = sum(t["games_played"] for t in all_team_stats.values())
    if total_games == 0:
        return 3.0  # valeur par défaut raisonnable pour le hockey si pas de données
    return total_goals / total_games


def compute_team_strengths(team_stats: dict, league_avg_goals: float) -> dict:
    """Force d'attaque et de défense d'une équipe, normalisées par la moyenne ligue.
    team_stats attendu : {"goals_for": int, "goals_against": int, "games_played": int}
    Force d'attaque > 1  -> attaque au-dessus de la moyenne
    Force de défense > 1 -> défense en-dessous de la moyenne (encaisse plus que la moyenne)"""
    games = team_stats.get("games_played", 0)
    if games == 0:
        return {"attack_strength": 1.0, "defense_strength": 1.0}
    avg_scored = team_stats["goals_for"] / games
    avg_conceded = team_stats["goals_against"] / games
    return {
        "attack_strength": avg_scored / league_avg_goals,
        "defense_strength": avg_conceded / league_avg_goals,
    }


# Valeurs calibrées par validation croisée (calibrées sur saison 2024, validées sur 2025
# le 26/08/2026 — voir model/calibrate.py et le journal de calibration) :
# home_advantage=1.09 / form_weight=0.0 / rest_penalty=0.0 → 56.9% précision, log-loss 0.856
# (form_weight=0 car la feature actuelle de forme récente n'améliorait pas le modèle)
HOME_ADVANTAGE = 1.09
FORM_WEIGHT = 0.0
REST_PENALTY = 0.0
H2H_WEIGHT = 0.10  # non utilisé activement pour l'instant, réservé pour une future version


def adjust_expected_goals(base_lambda: float, form_goal_diff_avg: float | None,
                           back_to_back: bool | None, is_home: bool,
                           form_weight: float = FORM_WEIGHT,
                           rest_penalty: float = REST_PENALTY) -> float:
    """Ajuste le lambda de base à partir des features contextuelles du match."""
    lam = base_lambda

    if form_goal_diff_avg is not None:
        # differentiel positif -> équipe en forme -> ajustement à la hausse
        lam *= (1 + form_weight * (form_goal_diff_avg / 3))  # /3 pour normaliser l'échelle

    if back_to_back:
        lam *= (1 - rest_penalty)

    return max(lam, 0.3)  # plancher pour éviter un lambda à 0 ou négatif


def compute_expected_goals(home_strengths: dict, away_strengths: dict,
                            league_avg_goals: float, features: dict,
                            home_advantage: float = HOME_ADVANTAGE,
                            form_weight: float = FORM_WEIGHT,
                            rest_penalty: float = REST_PENALTY) -> tuple[float, float]:
    """Calcule les lambdas (buts attendus) pour l'équipe à domicile et à l'extérieur.
    Les 3 poids (home_advantage, form_weight, rest_penalty) sont paramétrables
    pour permettre la calibration via model/calibrate.py — sinon les valeurs
    par défaut (non calibrées) sont utilisées."""
    home_lambda = (league_avg_goals * home_strengths["attack_strength"]
                   * away_strengths["defense_strength"] * home_advantage)
    away_lambda = (league_avg_goals * away_strengths["attack_strength"]
                   * home_strengths["defense_strength"])

    home_lambda = adjust_expected_goals(
        home_lambda, features.get("home_form_goal_diff_avg"),
        features.get("home_back_to_back"), is_home=True,
        form_weight=form_weight, rest_penalty=rest_penalty)
    away_lambda = adjust_expected_goals(
        away_lambda, features.get("away_form_goal_diff_avg"),
        features.get("away_back_to_back"), is_home=False,
        form_weight=form_weight, rest_penalty=rest_penalty)

    return round(home_lambda, 3), round(away_lambda, 3)


def score_probability_matrix(home_lambda: float, away_lambda: float,
                              max_goals: int = 10) -> list[list[float]]:
    """Matrice P(score_domicile=i, score_exterieur=j) pour i,j de 0 à max_goals."""
    home_probs = [poisson.pmf(i, home_lambda) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(j, away_lambda) for j in range(max_goals + 1)]
    return [[home_probs[i] * away_probs[j] for j in range(max_goals + 1)]
            for i in range(max_goals + 1)]


def summarize_predictions(matrix: list[list[float]]) -> dict:
    """Dérive toutes les prédictions utiles à partir de la matrice de scores."""
    n = len(matrix)
    home_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    draw = sum(matrix[i][j] for i in range(n) for j in range(n) if i == j)
    away_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)

    # Score exact le plus probable
    best_score, best_prob = (0, 0), 0.0
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > best_prob:
                best_prob = matrix[i][j]
                best_score = (i, j)

    # Over/Under sur quelques lignes courantes
    over_under = {}
    for line in (4.5, 5.5, 6.5, 7.5):
        over = sum(matrix[i][j] for i in range(n) for j in range(n) if i + j > line)
        over_under[f"over_{line}"] = round(over, 4)
        over_under[f"under_{line}"] = round(1 - over, 4)

    return {
        "home_win_pct": round(home_win, 4),
        "draw_pct": round(draw, 4),
        "away_win_pct": round(away_win, 4),
        "most_likely_score": f"{best_score[0]}-{best_score[1]}",
        "most_likely_score_pct": round(best_prob, 4),
        **over_under,
    }


def predict_match(home_stats: dict, away_stats: dict, league_avg_goals: float,
                   features: dict) -> dict:
    """Point d'entrée principal : combine tout pour prédire un match.

    home_stats / away_stats : {"goals_for": int, "goals_against": int, "games_played": int}
    features : sortie de build_prematch_features() ou build_match_features()
    """
    home_strengths = compute_team_strengths(home_stats, league_avg_goals)
    away_strengths = compute_team_strengths(away_stats, league_avg_goals)

    home_lambda, away_lambda = compute_expected_goals(
        home_strengths, away_strengths, league_avg_goals, features)

    matrix = score_probability_matrix(home_lambda, away_lambda)
    predictions = summarize_predictions(matrix)

    return {
        "home_expected_goals": home_lambda,
        "away_expected_goals": away_lambda,
        **predictions,
    }
