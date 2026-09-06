from scipy.stats import poisson

# Valeurs calibrées par validation croisée (calibrées sur saison 2024, validées sur 2025) :
# home_advantage=1.09 / form_weight=0.0 / rest_penalty=0.0 -> 56.9% précision, log-loss 0.856
# (form_weight=0 car la feature forme récente n'améliorait pas le modèle en test croisé)
HOME_ADVANTAGE = 1.09
FORM_WEIGHT = 0.0
REST_PENALTY = 0.0
H2H_BLEND_WEIGHT = 0.15
H2H_MIN_GAMES = 3


def compute_league_average(all_team_stats):
    total_goals = sum(t["goals_for"] for t in all_team_stats.values())
    total_games = sum(t["games_played"] for t in all_team_stats.values())
    return total_goals / total_games if total_games else 3.0


def compute_team_strengths(team_stats, league_avg_goals):
    games = team_stats.get("games_played", 0)
    if games == 0:
        return {"attack_strength": 1.0, "defense_strength": 1.0}
    avg_scored = team_stats["goals_for"] / games
    avg_conceded = team_stats["goals_against"] / games
    return {
        "attack_strength": avg_scored / league_avg_goals,
        "defense_strength": avg_conceded / league_avg_goals,
    }


def adjust_expected_goals(base_lambda, form_goal_diff_avg, back_to_back, is_home,
                           form_weight=FORM_WEIGHT, rest_penalty=REST_PENALTY):
    lam = base_lambda
    if form_goal_diff_avg is not None and form_weight:
        lam *= (1 + form_weight * (form_goal_diff_avg / 3))
    if back_to_back:
        lam *= (1 - rest_penalty)
    return max(lam, 0.3)


def compute_expected_goals(home_strengths, away_strengths, league_avg_goals, features,
                            home_advantage=HOME_ADVANTAGE, form_weight=FORM_WEIGHT,
                            rest_penalty=REST_PENALTY):
    home_lambda = (league_avg_goals * home_strengths["attack_strength"]
                   * away_strengths["defense_strength"] * home_advantage)
    away_lambda = (league_avg_goals * away_strengths["attack_strength"]
                   * home_strengths["defense_strength"])

    home_lambda = adjust_expected_goals(
        home_lambda, features.get("home_form_goal_diff_avg"),
        features.get("home_back_to_back"), True, form_weight, rest_penalty)
    away_lambda = adjust_expected_goals(
        away_lambda, features.get("away_form_goal_diff_avg"),
        features.get("away_back_to_back"), False, form_weight, rest_penalty)

    return round(home_lambda, 3), round(away_lambda, 3)


def score_probability_matrix(home_lambda, away_lambda, max_goals=10):
    home_probs = [poisson.pmf(i, home_lambda) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(j, away_lambda) for j in range(max_goals + 1)]
    return [[home_probs[i] * away_probs[j] for j in range(max_goals + 1)]
            for i in range(max_goals + 1)]


def summarize_predictions(matrix):
    n = len(matrix)
    home_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    draw = sum(matrix[i][j] for i in range(n) for j in range(n) if i == j)
    away_win = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)

    best_score, best_prob = (0, 0), 0.0
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > best_prob:
                best_prob = matrix[i][j]
                best_score = (i, j)

    over_under = {}
    for line in (3.5, 4.5, 5.5, 6.5, 7.5, 8.5):
        over = sum(matrix[i][j] for i in range(n) for j in range(n) if i + j > line)
        over_under[f"over_{line}"] = round(over, 4)
        over_under[f"under_{line}"] = round(1 - over, 4)

    puck_line = {}
    for handicap in (1.5, 2.5):
        home_covers = sum(matrix[i][j] for i in range(n) for j in range(n) if (i - j) > handicap)
        away_covers = sum(matrix[i][j] for i in range(n) for j in range(n) if (j - i) > handicap)
        puck_line[f"home_-{handicap}"] = round(home_covers, 4)
        puck_line[f"home_+{handicap}"] = round(1 - away_covers, 4)
        puck_line[f"away_-{handicap}"] = round(away_covers, 4)
        puck_line[f"away_+{handicap}"] = round(1 - home_covers, 4)

    double_chance = {
        "home_or_draw": round(home_win + draw, 4),
        "away_or_draw": round(away_win + draw, 4),
        "home_or_away": round(home_win + away_win, 4),
    }

    odd_total = sum(matrix[i][j] for i in range(n) for j in range(n) if (i + j) % 2 == 1)

    return {
        "home_win_pct": round(home_win, 4),
        "draw_pct": round(draw, 4),
        "away_win_pct": round(away_win, 4),
        "most_likely_score": f"{best_score[0]}-{best_score[1]}",
        "most_likely_score_pct": round(best_prob, 4),
        "over_under": over_under,
        "puck_line": puck_line,
        "double_chance": double_chance,
        "odd_even": {"even_total": round(1 - odd_total, 4), "odd_total": round(odd_total, 4)},
    }


def blend_with_h2h(pred, h2h_home_win_pct, h2h_games_count,
                    weight=H2H_BLEND_WEIGHT, min_games=H2H_MIN_GAMES):
    """Mélange léger avec l'historique des confrontations directes.
    N'agit que si assez de matchs d'historique existent, sinon renvoie pred inchangé."""
    if h2h_home_win_pct is None or h2h_games_count < min_games:
        return pred

    blended_home = (1 - weight) * pred["home_win_pct"] + weight * h2h_home_win_pct
    delta = blended_home - pred["home_win_pct"]
    blended_away = pred["away_win_pct"] - delta / 2
    blended_draw = pred["draw_pct"] - delta / 2

    pred = dict(pred)
    pred["home_win_pct"] = round(max(blended_home, 0), 4)
    pred["away_win_pct"] = round(max(blended_away, 0), 4)
    pred["draw_pct"] = round(max(blended_draw, 0), 4)
    return pred
