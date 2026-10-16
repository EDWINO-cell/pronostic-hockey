import streamlit as st
from datetime import datetime, timezone, timedelta

from highlightly import HighlightlyClient
from poisson import (
    compute_league_average, compute_team_strengths,
    compute_expected_goals, score_probability_matrix, summarize_predictions,
    blend_with_h2h,
)
from xg_model import (
    get_live_team_xg_stats, remap_stats_to_highlightly_ids,
    predict_team_sog, LEAGUE_AVG_SOG, LEAGUE_STD_SOG,
)
from scipy.stats import norm

st.set_page_config(page_title="Prédictions NHL", page_icon="🏒", layout="wide")

CURRENT_SEASON = 2027          # saison Highlightly en cours (équipes, matchs à venir, h2h)
                                # NB: Highlightly numérote par année DE FIN de saison
                                # (ex: saison 2026-2027 = season 2027, pas 2026)
NHL_SEASON_ID = "20262027"     # saison NHL officielle en cours (xG, format AAAAAAAA)


@st.cache_resource
def get_client():
    return HighlightlyClient(api_key=st.secrets.get("HIGHLIGHTLY_API_KEY"))


@st.cache_data(ttl=86400)
def load_teams():
    return get_client().get_teams(league="NHL")


@st.cache_data(ttl=86400)
def load_team_stats(_teams):
    """Stats d'équipe basées sur le xG (API NHL officielle), plus fiables que les
    buts bruts selon le backtest (54.8% vs 53.3% de précision). Si le xG n'a pas
    encore assez de matchs pour une équipe (ex: tout début de saison), on retombe
    sur les buts Highlightly pour cette équipe précise."""
    xg_stats = get_live_team_xg_stats(NHL_SEASON_ID)
    xg_by_highlightly_id = remap_stats_to_highlightly_ids(xg_stats, _teams)

    fallback_stats = get_client().get_all_team_stats(year=CURRENT_SEASON, league_type="NHL")

    merged = dict(fallback_stats)
    merged.update(xg_by_highlightly_id)  # le xG écrase le fallback quand disponible
    return merged


def get_sog_stats(team_stats_entry: dict) -> dict:
    """Extrait les moyennes de tirs cadrés d'une entrée de team_stats (si
    disponible via le xG NHL ; sinon valeurs moyenne ligue par défaut)."""
    games = team_stats_entry.get("games_played", 0)
    if games == 0 or "sog_for" not in team_stats_entry:
        return {"sog_for_avg": LEAGUE_AVG_SOG, "sog_against_avg": LEAGUE_AVG_SOG}
    return {
        "sog_for_avg": team_stats_entry["sog_for"] / games,
        "sog_against_avg": team_stats_entry["sog_against"] / games,
    }


def predict_sog_markets(home_stats_entry: dict, away_stats_entry: dict) -> dict:
    """Prédictions over/under tirs cadrés : total du match (validé 67.8%) et
    par équipe séparée (validé 61.4%). Loi normale (variance >> moyenne en NHL)."""
    home_sog_stats = get_sog_stats(home_stats_entry)
    away_sog_stats = get_sog_stats(away_stats_entry)

    pred_home_sog = predict_team_sog(home_sog_stats, away_sog_stats)
    pred_away_sog = predict_team_sog(away_sog_stats, home_sog_stats)
    pred_total_sog = pred_home_sog + pred_away_sog

    std_total = LEAGUE_STD_SOG * (2 ** 0.5)  # approximation : deux équipes ~indépendantes

    return {
        "home_sog_pred": round(pred_home_sog, 1),
        "away_sog_pred": round(pred_away_sog, 1),
        "total_sog_pred": round(pred_total_sog, 1),
        "total_over_58.5": round(1 - norm.cdf(58.5, loc=pred_total_sog, scale=std_total), 4),
        "home_over_28.5": round(1 - norm.cdf(28.5, loc=pred_home_sog, scale=LEAGUE_STD_SOG), 4),
        "away_over_28.5": round(1 - norm.cdf(28.5, loc=pred_away_sog, scale=LEAGUE_STD_SOG), 4),
    }


@st.cache_data(ttl=86400)
def load_upcoming_matches(season, days_ahead=14):
    """Interroge jour par jour plutôt que de se fier au tri par défaut de
    /matches avec un gros 'limit' : testé et confirmé plus fiable (le tri
    par défaut de l'API n'est pas garanti chronologique)."""
    client = get_client()
    today = datetime.now(timezone.utc).date()
    upcoming = []
    for offset in range(days_ahead):
        day = today + timedelta(days=offset)
        day_str = day.isoformat()
        result = client.get_matches(league="NHL", season=season, date_str=day_str, limit=20)
        for m in result.get("data", []):
            if m.get("state", {}).get("report") == "Scheduled":
                upcoming.append(m)
    upcoming.sort(key=lambda m: m["date"])
    return upcoming


@st.cache_data(ttl=86400)
def load_features(home_team_id, away_team_id):
    client = get_client()

    def _recent_form(team_id):
        recent = client.get_team_recent_matches(team_id, league="NHL", season=CURRENT_SEASON, n=10)
        if not recent:
            return None
        diff_total = 0
        for m in recent:
            hs, aws = (int(x) for x in m["state"]["score"]["current"].split(" - "))
            is_home = m["homeTeam"]["id"] == team_id
            diff_total += (hs - aws) if is_home else (aws - hs)
        return round(diff_total / len(recent), 2)

    h2h = client.get_head_to_head(home_team_id, away_team_id, league="NHL", limit=50)
    h2h_matches = [m for m in h2h.get("data", []) if m.get("state", {}).get("report") == "Final"]
    h2h_home_wins, h2h_total = 0, 0
    for m in h2h_matches:
        hs, aws = (int(x) for x in m["state"]["score"]["current"].split(" - "))
        winner_is_match_home = hs > aws
        winner_id = m["homeTeam"]["id"] if winner_is_match_home else m["awayTeam"]["id"]
        if winner_id == home_team_id:
            h2h_home_wins += 1
        h2h_total += 1

    return {
        "home_form_goal_diff_avg": _recent_form(home_team_id),
        "away_form_goal_diff_avg": _recent_form(away_team_id),
        "h2h_games_count": h2h_total,
        "h2h_home_win_pct": round(h2h_home_wins / h2h_total, 3) if h2h_total else None,
    }


def predict(home_team_id, away_team_id, team_stats, features):
    league_avg = compute_league_average(team_stats)
    home_stats = team_stats.get(home_team_id, {"goals_for": 0, "goals_against": 0, "games_played": 0})
    away_stats = team_stats.get(away_team_id, {"goals_for": 0, "goals_against": 0, "games_played": 0})

    home_strengths = compute_team_strengths(home_stats, league_avg)
    away_strengths = compute_team_strengths(away_stats, league_avg)

    home_lambda, away_lambda = compute_expected_goals(
        home_strengths, away_strengths, league_avg, features)

    matrix = score_probability_matrix(home_lambda, away_lambda)
    pred = summarize_predictions(matrix)
    pred = blend_with_h2h(pred, features.get("h2h_home_win_pct"), features.get("h2h_games_count", 0))
    return pred


def render_prediction(home_name, away_name, pred, features, sog_pred=None):
    col1, col2, col3 = st.columns(3)
    col1.metric(f"🏠 {home_name}", f"{pred['home_win_pct']*100:.1f}%")
    col2.metric("Nul", f"{pred['draw_pct']*100:.1f}%")
    col3.metric(f"✈️ {away_name}", f"{pred['away_win_pct']*100:.1f}%")

    st.caption(
        f"Score le plus probable : **{pred['most_likely_score']}** "
        f"({pred['most_likely_score_pct']*100:.1f}%)"
    )

    if features.get("h2h_games_count"):
        st.caption(
            f"Face-à-face ({features['h2h_games_count']} matchs) : "
            f"{home_name} gagne {features['h2h_home_win_pct']*100:.0f}% du temps à domicile"
        )
    form_h = features.get("home_form_goal_diff_avg")
    form_a = features.get("away_form_goal_diff_avg")
    if form_h is not None or form_a is not None:
        st.caption(
            f"Forme récente (diff. de buts/match, 10 derniers) — "
            f"{home_name}: {form_h if form_h is not None else 'N/A'} · "
            f"{away_name}: {form_a if form_a is not None else 'N/A'}"
        )

    with st.expander("Autres marchés"):
        tabs_list = ["Over/Under", "Handicap", "Double chance", "Pair/Impair"]
        if sog_pred:
            tabs_list.append("Tirs cadrés")
        all_tabs = st.tabs(tabs_list)
        ou_tab, pl_tab, dc_tab, oe_tab = all_tabs[:4]
        sog_tab = all_tabs[4] if sog_pred else None

        with ou_tab:
            for line in (3.5, 4.5, 5.5, 6.5, 7.5, 8.5):
                st.write(f"Over {line} : {pred['over_under'][f'over_{line}']*100:.1f}% "
                         f"| Under {line} : {pred['over_under'][f'under_{line}']*100:.1f}%")

        with pl_tab:
            for handicap in (1.5, 2.5):
                st.write(f"{home_name} -{handicap} : {pred['puck_line'][f'home_-{handicap}']*100:.1f}% "
                         f"| {away_name} +{handicap} : {pred['puck_line'][f'away_+{handicap}']*100:.1f}%")
                st.write(f"{away_name} -{handicap} : {pred['puck_line'][f'away_-{handicap}']*100:.1f}% "
                         f"| {home_name} +{handicap} : {pred['puck_line'][f'home_+{handicap}']*100:.1f}%")

        with dc_tab:
            st.write(f"{home_name} ou nul : {pred['double_chance']['home_or_draw']*100:.1f}%")
            st.write(f"{away_name} ou nul : {pred['double_chance']['away_or_draw']*100:.1f}%")
            st.write(f"{home_name} ou {away_name} (pas de nul) : {pred['double_chance']['home_or_away']*100:.1f}%")

        with oe_tab:
            st.write(f"Total pair : {pred['odd_even']['even_total']*100:.1f}%")
            st.write(f"Total impair : {pred['odd_even']['odd_total']*100:.1f}%")

        if sog_tab:
            with sog_tab:
                st.caption("Modèle validé par backtest : 67.8% (total du match), 61.4% (par équipe)")
                st.write(f"Total tirs cadrés prédit : **{sog_pred['total_sog_pred']}** "
                         f"(Over 58.5 : {sog_pred['total_over_58.5']*100:.1f}%)")
                st.write(f"{home_name} : **{sog_pred['home_sog_pred']}** tirs prédits "
                         f"(Over 28.5 : {sog_pred['home_over_28.5']*100:.1f}%)")
                st.write(f"{away_name} : **{sog_pred['away_sog_pred']}** tirs prédits "
                         f"(Over 28.5 : {sog_pred['away_over_28.5']*100:.1f}%)")


st.title("🏒 Prédictions NHL")
st.caption(
    "Modèle Poisson avec xG (API NHL officielle) + face-à-face. "
    "Précision attendue sur l'issue du match : ~55%. Ne constitue pas un conseil de pari."
)

tab_upcoming, tab_manual = st.tabs(["📅 Matchs à venir", "🔄 Comparer deux équipes"])

try:
    teams = load_teams()
    teams_by_id = {t["id"]: t for t in teams if t.get("logo")}
    team_stats = load_team_stats(teams)
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    st.stop()

with tab_upcoming:
    upcoming = load_upcoming_matches(CURRENT_SEASON)
    if not upcoming:
        st.info("Aucun match programmé dans les 14 prochains jours pour le moment "
                "(hors-saison ou calendrier pas encore publié).")
    else:
        for m in upcoming:
            home, away = m["homeTeam"], m["awayTeam"]
            match_date = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
            st.subheader(f"{home['displayName']} vs {away['displayName']}")
            st.caption(match_date.strftime("%d %B %Y"))
            pred = predict(home["id"], away["id"], team_stats, features={})
            home_stats_entry = team_stats.get(home["id"], {})
            away_stats_entry = team_stats.get(away["id"], {})
            sog_pred = predict_sog_markets(home_stats_entry, away_stats_entry)
            render_prediction(home["displayName"], away["displayName"], pred, features={}, sog_pred=sog_pred)
            st.divider()

with tab_manual:
    team_options = {t["displayName"]: t["id"] for t in teams_by_id.values()}
    sorted_names = sorted(team_options.keys())

    col1, col2 = st.columns(2)
    home_name = col1.selectbox("Équipe à domicile", sorted_names, index=0)
    away_name = col2.selectbox("Équipe à l'extérieur", sorted_names, index=1)

    if home_name == away_name:
        st.warning("Choisis deux équipes différentes.")
    elif st.button("Prédire ce match", type="primary"):
        features = load_features(team_options[home_name], team_options[away_name])
        pred = predict(team_options[home_name], team_options[away_name], team_stats, features)
        home_stats_entry = team_stats.get(team_options[home_name], {})
        away_stats_entry = team_stats.get(team_options[away_name], {})
        sog_pred = predict_sog_markets(home_stats_entry, away_stats_entry)
        render_prediction(home_name, away_name, pred, features, sog_pred=sog_pred)
