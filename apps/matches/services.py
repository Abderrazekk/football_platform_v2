"""
Role  : Load match classification artifacts and predict H/D/A probabilities.
Input : Home team name, away team name.
Output: Dictionary with probabilities (sum = 100%), predicted result, and model version.
"""


def predict_match(home_team, away_team):
    """Stub prediction function. Will be replaced by joblib.load() in Phase 4."""
    return {
        "home_team": home_team,
        "away_team": away_team,
        "p_home": 45.0,
        "p_draw": 25.0,
        "p_away": 30.0,
        "predicted_result": "H",
        "model_version": "stub_v0",
    }