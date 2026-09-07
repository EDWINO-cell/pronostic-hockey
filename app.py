import streamlit as st
from datetime import datetime, timezone, timedelta

from highlightly import HighlightlyClient
from poisson import (
    compute_league_average, compute_team_strengths,
    compute_expected_goals, score_probability_matrix, summarize_predictions,
    blend_with_h2h,
)

st.set_page_config(page_title="Prédictions NHL", page_icon="🏒", layout="wide")

CURRENT_SEASON = 2025


@st.cache_resource
def get_client():
    return HighlightlyClient(api_key=st.secrets.get("HIGHLIGHTLY_API_KEY"))


@st.cache_data(ttl=86400)
def load_teams():
    return get_client().get_teams(league="NHL")


@st.cache_data(ttl=86400)
def load_team_stats():
    return get_client().get_all_team_stats(year=CURRENT_SEASON, league_type="NHL")


@st.cache_data(ttl=86400)
def load_upcoming_matches(days_ahead=14):
    client = get_client()
    matches = client.get_matches(league="NHL", season=CURRENT_SEASON, limit=100)
    all_matches = matches.get("data", [])
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=days_ahead)
    upcoming = [
        m for m in all_matches
        if m.get("state", {}).get("report") == "Scheduled"
        and now <= datetime.fromisoformat(m["date"].replace("Z", "+00:00")) <= horizon
    ]
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


def render_prediction(home_name, away_name, pred, features):
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
        ou_tab, pl_tab, dc_tab, oe_tab = st.tabs(["Over/Under", "Handicap", "Double chance", "Pair/Impair"])

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


st.title("🏒 Prédictions NHL")
st.caption(
    "Modèle Poisson calibré par validation croisée + face-à-face. "
    "Précision attendue sur l'issue du match : ~57%. Ne constitue pas un conseil de pari."
)

tab_upcoming, tab_manual = st.tabs(["📅 Matchs à venir", "🔄 Comparer deux équipes"])

try:
    teams = load_teams()
    teams_by_id = {t["id"]: t for t in teams if t.get("logo")}
    team_stats = load_team_stats()
except Exception as e:
    st.error(f"Erreur lors du chargement des données Highlightly : {e}")
    st.stop()

with tab_upcoming:
    upcoming = load_upcoming_matches()
    if not upcoming:
        st.info("Aucun match programmé dans les 14 prochains jours pour le moment "
                "(hors-saison ou calendrier pas encore publié).")
    else:
        for m in upcoming:
            home, away = m["homeTeam"], m["awayTeam"]
            match_date = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
            st.subheader(f"{home['displayName']} vs {away['displayName']}")
            st.caption(match_date.strftime("%d %B %Y"))
            # Pas de forme récente / face-à-face ici : en pleine saison, le nombre de
            # matchs à venir peut dépasser le quota API si on calcule ces features pour
            # chacun automatiquement. Réservé au comparateur manuel (à la demande).
            pred = predict(home["id"], away["id"], team_stats, features={})
            render_prediction(home["displayName"], away["displayName"], pred, features={})
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
        render_prediction(home_name, away_name, pred, features)
    
