# app_v8_api_fallback_prop_model.py
# NFL Player Prop Model – API version with fallback for players and teams

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import requests
import re

st.set_page_config(page_title="NFL Player Prop Model (API-Sports v8)", layout="centered")

# --------------------------------
# 1) Your API Key
# --------------------------------
API_KEY = "84ac3b6a25159b2ed86719caf4eaf776"
HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://v1.american-football.api-sports.io"

# --------------------------------
# 2) Helper Functions
# --------------------------------
def normalize_header(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

@st.cache_data(show_spinner=False)
def get_players():
    try:
        url = f"{BASE_URL}/players?league=1&season=2025"
        res = requests.get(url, headers=HEADERS, timeout=15)
        data = res.json()
        players = []
        for item in data.get("response", []):
            name = item.get("player", {}).get("name", "")
            team = item.get("team", {}).get("name", "")
            if name:
                players.append({"player": name, "team": team})
        return pd.DataFrame(players)
    except Exception as e:
        st.warning(f"⚠️ Could not load players from API: {e}")
        return pd.DataFrame(columns=["player", "team"])

@st.cache_data(show_spinner=False)
def get_teams():
    try:
        url = f"{BASE_URL}/teams?league=1&season=2025"
        res = requests.get(url, headers=HEADERS, timeout=15)
        data = res.json()
        teams = []
        for t in data.get("response", []):
            name = t.get("name", "")
            if name:
                teams.append({"team": name})
        return pd.DataFrame(teams)
    except Exception as e:
        st.warning(f"⚠️ Could not load teams from API: {e}")
        return pd.DataFrame(columns=["team"])

# --------------------------------
# 3) Load Data
# --------------------------------
players_df = get_players()
teams_df = get_teams()

# --- Fallback for players ---
if "player" in players_df.columns and not players_df.empty:
    player_list = sorted(players_df["player"].dropna().unique().tolist())
else:
    st.warning("⚠️ NFL players could not be loaded from the API. Using fallback sample list.")
    player_list = [
        "Patrick Mahomes", "Josh Allen", "Joe Burrow", "Jalen Hurts",
        "Christian McCaffrey", "Derrick Henry", "Tyreek Hill", "Justin Jefferson",
        "CeeDee Lamb", "Amon-Ra St. Brown", "Travis Kelce", "Davante Adams",
        "Lamar Jackson", "Brock Purdy", "Breece Hall", "Stefon Diggs"
    ]

# --- Fallback for teams ---
if "team" in teams_df.columns and not teams_df.empty:
    team_list = sorted(teams_df["team"].dropna().unique().tolist())
else:
    st.warning("⚠️ NFL teams could not be loaded from the API. Using fallback list.")
    team_list = [
        "Buffalo Bills", "Miami Dolphins", "New England Patriots", "New York Jets",
        "Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns", "Pittsburgh Steelers",
        "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Tennessee Titans",
        "Denver Broncos", "Kansas City Chiefs", "Las Vegas Raiders", "Los Angeles Chargers",
        "Dallas Cowboys", "New York Giants", "Philadelphia Eagles", "Washington Commanders",
        "Chicago Bears", "Detroit Lions", "Green Bay Packers", "Minnesota Vikings",
        "Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers",
        "Arizona Cardinals", "Los Angeles Rams", "San Francisco 49ers", "Seattle Seahawks"
    ]

# --------------------------------
# 4) UI Dropdowns
# --------------------------------
st.title("🏈 NFL Player Prop Model (API + Fallback v8)")
player_name = st.selectbox("Select Player:", [""] + player_list, index=0)
opponent_team = st.selectbox("Select Opponent Team:", [""] + team_list, index=0)

prop_choices = [
    "passing_yards", "rushing_yards", "receiving_yards",
    "receptions", "targets", "carries", "anytime_td"
]
selected_props = st.multiselect("Select props:", prop_choices, default=["receiving_yards"])

lines = {}
for prop in selected_props:
    if prop != "anytime_td":
        lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=prop)

if not player_name or not opponent_team or not selected_props:
    st.stop()

st.header("📊 Results")

# --------------------------------
# 5) Dummy Prop Logic (placeholder for your working model)
# --------------------------------
for prop in selected_props:
    if prop == "anytime_td":
        st.subheader("🔥 Anytime TD (Rushing + Receiving + Defense Adjusted)")
        st.write(f"This would calculate TD odds for **{player_name}** vs **{opponent_team}**.")
        continue

    line_val = lines.get(prop, 0.0)
    predicted_pg = np.random.uniform(line_val * 0.8, line_val * 1.2)
    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = 1 - norm.cdf(z)
    prob_under = norm.cdf(z)
    prob_over = float(np.clip(prob_over, 0.001, 0.999))
    prob_under = float(np.clip(prob_under, 0.001, 0.999))

    st.subheader(prop.replace("_", " ").title())
    st.write(f"**Predicted (this game):** {predicted_pg:.2f}")
    st.write(f"**Line:** {line_val}")
    st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
    st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

    fig_bar = px.bar(x=["Predicted (this game)", "Line"], y=[predicted_pg, line_val],
                     title=f"{player_name} – {prop.replace('_', ' ').title()}")
    st.plotly_chart(fig_bar, use_container_width=True)
