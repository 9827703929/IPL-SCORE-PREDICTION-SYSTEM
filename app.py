from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import joblib
from models.predictor import CricketPredictor

app = Flask(__name__)
CORS(app)

# ─── API KEYS (replace with your real keys) ───────────────────────────────────
CRICAPI_KEY = os.environ.get("CRICAPI_KEY", "YOUR_CRICAPI_KEY_HERE")
CRICAPI_BASE = "https://api.cricapi.com/v1"

predictor = CricketPredictor()

# ─── HELPER: CricAPI requests ─────────────────────────────────────────────────
def cricapi_get(endpoint, params=None):
    if params is None:
        params = {}
    params["apikey"] = CRICAPI_KEY
    try:
        r = requests.get(f"{CRICAPI_BASE}/{endpoint}", params=params, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"API Error: {e}")
        return None

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/live-matches")
def live_matches():
    # Check if API key is set
    if CRICAPI_KEY == "YOUR_CRICAPI_KEY_HERE":
        return jsonify({
            "status": "no_key",
            "matches": [],
            "message": "No API key set. Get a free key at cricapi.com and add it to your .env file as CRICAPI_KEY=your_key_here"
        })

    data = cricapi_get("currentMatches", {"offset": 0})
    if data and data.get("status") == "success":
        matches = data.get("data", [])
        live = [m for m in matches if m.get("matchStarted") and not m.get("matchEnded")]
        if not live:
            # Also include recently finished matches so the section is not empty
            recent = [m for m in matches if m.get("matchStarted")][:5]
            return jsonify({"status": "success", "matches": recent, "note": "No live matches right now — showing recent matches"})
        return jsonify({"status": "success", "matches": live[:10]})

    return jsonify({
        "status": "api_error",
        "matches": [],
        "message": "Could not reach CricAPI. Check your API key or internet connection."
    })

@app.route("/api/match-detail/<match_id>")
def match_detail(match_id):
    data = cricapi_get("match_info", {"id": match_id})
    if data and data.get("status") == "success":
        return jsonify({"status": "success", "data": data.get("data", {})})
    return jsonify({"status": "error", "message": "Match not found"})

@app.route("/api/predict-score", methods=["POST"])
def predict_score():
    body = request.json
    try:
        result = predictor.predict_final_score(
            team=body.get("batting_team", ""),
            opponent=body.get("bowling_team", ""),
            stadium=body.get("venue", ""),
            current_runs=int(body.get("current_runs", 0)),
            current_wickets=int(body.get("current_wickets", 0)),
            current_over=float(body.get("current_over", 1)),
            total_overs=int(body.get("total_overs", 20)),
            players=body.get("players", [])
        )
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/predict-winner", methods=["POST"])
def predict_winner():
    body = request.json
    try:
        result = predictor.predict_win_probability(
            batting_team=body.get("batting_team", ""),
            bowling_team=body.get("bowling_team", ""),
            venue=body.get("venue", ""),
            target=int(body.get("target", 0)),
            current_runs=int(body.get("current_runs", 0)),
            current_wickets=int(body.get("current_wickets", 0)),
            current_over=float(body.get("current_over", 1)),
            total_overs=int(body.get("total_overs", 20)),
            players=body.get("players", [])
        )
        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/teams")
def get_teams():
    return jsonify({"teams": predictor.get_team_list()})

@app.route("/api/venues")
def get_venues():
    return jsonify({"venues": predictor.get_venue_list()})

@app.route("/api/players/<team>")
def get_players(team):
    return jsonify({"players": predictor.get_player_list(team)})

@app.route("/api/model-stats")
def model_stats():
    return jsonify(predictor.get_model_statistics())


if __name__ == "__main__":
       app.run(host='0.0.0.0', port=5000, debug=True)
