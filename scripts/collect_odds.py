"""Collecte quotidienne des cotes NHL (The Odds API) -> data/odds_log.json

Lancé automatiquement par GitHub Actions (voir .github/workflows/collect_odds.yml).
La clé API est lue dans la variable d'environnement ODDS_API_KEY (secret GitHub),
jamais écrite dans le dépôt (qui est public).
"""
import json
import os
import sys
from datetime import datetime, timezone

import requests

LOG_PATH = "data/odds_log.json"
API_URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"


def devig_two_way(price_a, price_b):
    """Cotes décimales -> probabilités sans la marge du bookmaker."""
    ia, ib = 1 / price_a, 1 / price_b
    total = ia + ib
    return ia / total, ib / total


def average_devigged(bookmakers, home, away):
    home_probs, away_probs = [], []
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in market["outcomes"]}
            if home in prices and away in prices:
                h, a = devig_two_way(prices[home], prices[away])
                home_probs.append(h)
                away_probs.append(a)
    if not home_probs:
        return None, None
    return sum(home_probs) / len(home_probs), sum(away_probs) / len(away_probs)


def main():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        sys.exit("ODDS_API_KEY manquante (secret GitHub à créer)")

    resp = requests.get(
        API_URL,
        params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=30,
    )
    resp.raise_for_status()
    print("Crédits restants ce mois :", resp.headers.get("x-requests-remaining"))
    games = resp.json()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {"entries": []}

    now = datetime.now(timezone.utc)
    added = 0
    for g in games:
        commence = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        if commence <= now:
            continue  # match déjà commencé : on ignore les cotes en direct
        home_p, away_p = average_devigged(g["bookmakers"], g["home_team"], g["away_team"])
        if home_p is None:
            continue
        log["entries"].append({
            "event_id": g["id"],
            "collected_at": now.isoformat(timespec="seconds"),
            "collected_date": now.date().isoformat(),
            "commence_time": g["commence_time"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "market_home_prob": round(home_p, 4),
            "market_away_prob": round(away_p, 4),
            "n_bookmakers": len(g["bookmakers"]),
            "our_home_prob": None,
            "actual_result": None,
        })
        added += 1

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1, ensure_ascii=False)
    print(f"{added} entrées ajoutées, {len(log['entries'])} au total")


if __name__ == "__main__":
    main()
      
