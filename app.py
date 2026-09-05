"""
App Streamlit de prédiction de matchs NHL.

Deux vues :
  - Matchs à venir : liste des prochains matchs avec prédictions automatiques
  - Comparer deux équipes : sélecteur manuel pour simuler n'importe quel affrontement

Les données Highlightly sont mises en cache 24h (st.cache_data) pour respecter
la limite de 100 requêtes/jour du plan gratuit — une seule actualisation par
jour, peu importe le nombre de visiteurs.
"""
import streamlit as st
from datetime import datetime, timezone, timedelta

from highlightly import HighlightlyClient
from poisson import (
    compute_league_average, compute_team_strengths,
    compute_expected_goals, score_probability_matrix, summarize_predictions,
)

st.set_page_config(page_title="Prédictions NHL", page_icon="🏒", layout="wide")

CURRENT_SEASON = 2025  # à ajuster manuellement au démarrage de la saison 2026-2027


@st.cache_resource
def get_client() -> HighlightlyClient:
    api_key = st.secrets.get("HIGHLIGHTLY_API_KEY")
    return HighlightlyClient(api_key=api_key)


@st.cache_data(ttl=86400)  # 24h — une seule actualisation par jour
def load_teams():
    client = get_client()
    return client.get_teams(league="NHL")


@st.cache_data(ttl=86400)
def load_team_stats():
    client = get_client()
    return client.get_all_team_stats(year=CURRENT_SEASON, league_type="NHL")


@st.cache_data(ttl=86400)
def load_upcoming_matches(days_ahead: int = 14):
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


def predict(home_team_id: int, away_team_id: int, team_stats: dict) -> dict:
    """Prédiction sans features contextuelles (forme/repos) pour l'app —
    celles-ci nécessiteraient des appels API supplémentaires par match,
    ce qu'on évite pour rester dans le quota. Basé uniquement sur les
    forces d'attaque/défense saison, calibrées par model/calibrate.py."""
    league_avg = compute_league_average(team_stats)
    home_stats = team_stats.get(home_team_id, {"goals_for": 0, "goals_against": 0, "games_played": 0})
    away_stats = team_stats.get(away_team_id, {"goals_for": 0, "goals_against": 0, "games_played": 0})

    home_strengths = compute_team_strengths(home_stats, league_avg)
    away_strengths = compute_team_strengths(away_stats, league_avg)

    home_lambda, away_lambda = compute_expected_goals(
        home_strengths, away_strengths, league_avg, features={})

    matrix = score_probability_matrix(home_lambda, away_lambda)
    return summarize_predictions(matrix)


def render_prediction(home_name: str, away_name: str, pred: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric(f"🏠 {home_name}", f"{pred['home_win_pct']*100:.1f}%")
    col2.metric("Nul", f"{pred['draw_pct']*100:.1f}%")
    col3.metric(f"✈️ {away_name}", f"{pred['away_win_pct']*100:.1f}%")

    st.caption(
        f"Score le plus probable : **{pred['most_likely_score']}** "
        f"({pred['most_likely_score_pct']*100:.1f}% de chance) — "
        f"Over 5.5 buts : {pred['over_5.5']*100:.1f}%"
    )


# --- Interface ---
st.title("🏒 Prédictions NHL")
st.caption(
    "Modèle statistique Poisson calibré par validation croisée (saisons 2024/2025). "
    "Précision attendue sur l'issue du match : ~57%. Ne constitue pas un conseil de pari."
)

tab_upcoming, tab_manual = st.tabs(["📅 Matchs à venir", "🔄 Comparer deux équipes"])

try:
    teams = load_teams()
    teams_by_id = {t["id"]: t for t in teams if t.get("logo")}  # exclut divisions/conférences
    team_stats = load_team_stats()
except Exception as e:
    st.error(f"Erreur lors du chargement des données Highlightly : {e}")
    st.stop()

with tab_upcoming:
    upcoming = load_upcoming_matches()
    if not upcoming:
        st.info(
            "Aucun match programmé dans les 14 prochains jours pour le moment "
            "(hors-saison ou calendrier pas encore publié)."
        )
    else:
        for m in upcoming:
            home = m["homeTeam"]
            away = m["awayTeam"]
            match_date = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
            st.subheader(f"{home['displayName']} vs {away['displayName']}")
            st.caption(match_date.strftime("%d %B %Y"))
            pred = predict(home["id"], away["id"], team_stats)
            render_prediction(home["displayName"], away["displayName"], pred)
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
        pred = predict(team_options[home_name], team_options[away_name], team_stats)
        render_prediction(home_name, away_name, pred)
