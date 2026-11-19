# FILE   : ml/predict.py
# ROLE   : public ML entry point used by Django (STUB in Phase 0)
# INPUT  : team names, player names, season
# OUTPUT : dicts / lists with fake but valid values
# RUN    : not executed directly; imported from Django services

# TODO (Phase 4): replace bodies with real model loading
# from ml/artifacts/models/*.joblib


def predict_match(home_team: str, away_team: str) -> dict:
    """Return fake probabilities for H / D / A (sum = 1.0)."""
    return {
        "home_win": 0.45,
        "draw": 0.25,
        "away_win": 0.30,
    }


def predict_player_points(player_name: str, opponent: str, was_home: bool = True) -> dict:
    """Return fake expected points for a player in a match."""
    return {
        "player": player_name,
        "opponent": opponent,
        "expected_points": 4.0,
        "uncertainty": 2.0,
    }


def get_best_xi(season: str = "2026-27") -> list:
    """Return a fake list of 11 players for the best XI."""
    return [f"Player_{i}" for i in range(1, 12)]