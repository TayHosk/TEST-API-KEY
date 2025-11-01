# app_v8_nfl_api_prop_model.py
# NFL version of your working v7.7 app
# - uses API-Sports NFL (American Football) instead of Google Sheets
# - keeps: player dropdown, team dropdown, multi-prop, anytime TD (rush+rec)
# - builds synthetic def tables from team stats so your old logic still works

import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from scipy.stats import norm
import plotly.express as px

st.set_page_config(page_title="NFL Player Prop Model (API-Sports v8)", layout="centered")

# -------------------------------------------------------
# 0) YOUR API KEY (you gave it to me)
# -------------------------------------------------------
API_KEY = "84ac3b6a25159b2ed86719caf4eaf776"
BASE_URL = "https://v1.american-football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# -------------------------------------------------------
# 1) small helpers
# -------------------------------------------------------
def normalize_header(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

def safe_div(a, b):
    if b is None or b == 0:
        return 0.0
    return a / b

# -------------------------------------------------------
# 2) fetch leagues → pick NFL
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_nfl_league_id():
    # API-Sports NFL doc: /leagues returns all; find NFL
    url = f"{BASE_URL}/leagues"
    r = requests.get(url, headers=HEADERS)
    data = r.json()
    # try to find "NFL" in name
    for item in data.get("response", []):
        lg = item.get("league", {})
        name = lg.get("name", "")
        if "nfl" in name.lower():
            return lg.get("id")
    # fallback: 1
    return 1

NFL_LEAGUE_ID = fetch_nfl_league_id()

# -------------------------------------------------------
# 3) fetch teams for NFL season
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_nfl_teams(season: int = 2025):
    url = f"{BASE_URL}/teams?league={NFL_LEAGUE_ID}&season={season}"
    r = requests.get(url, headers=HEADERS)
    data = r.json()
    teams = []
    for item in data.get("response", []):
        team_info = item.get("team", {})
        name = team_info.get("name")
        team_id = team_info.get("id")
        if name and team_id:
            teams.append({"team_id": team_id, "team": name})
    return pd.DataFrame(teams)

# -------------------------------------------------------
# 4) fetch players for ONE team
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_players_for_team(team_id: int, season: int = 2025):
    # API is paginated sometimes; keep it simple
    url = f"{BASE_URL}/players?league={NFL_LEAGUE_ID}&season={season}&team={team_id}"
    r = requests.get(url, headers=HEADERS)
    data = r.json()
    rows = []
    for item in data.get("response", []):
        player = item.get("player", {})
        stats_list = item.get("statistics", [])
        # statistics might be empty, we still want player name
        name = player.get("name", "")
        pos = player.get("position", "")
        team_name = ""
        games_played = 0
        rush_yds = 0
        rec_yds = 0
        pass_yds = 0
        rec_tds = 0
        rush_tds = 0
        pass_tds = 0

        if stats_list:
            stats = stats_list[0]
            team_name = stats.get("team", {}).get("name", "")
            games_played = stats.get("games", {}).get("played", 0)

            # API-Sports american-football: stats often nested by "passing", "rushing", "receiving"
            passing = stats.get("passing", {})
            rushing = stats.get("rushing", {})
            receiving = stats.get("receiving", {})
            scoring = stats.get("scoring", {})

            pass_yds = passing.get("yards", 0) or 0
            pass_tds = passing.get("touchdowns", 0) or 0

            rush_yds = rushing.get("yards", 0) or 0
            rush_tds = rushing.get("touchdowns", 0) or 0

            rec_yds = receiving.get("yards", 0) or 0
            rec_tds = receiving.get("touchdowns", 0) or 0

            # if scoring exists, could also check total TDs

        rows.append({
            "player": name,
            "team": team_name,
            "position": pos.lower() if pos else "",
            "games_played": games_played or 1,
            "passing_yards_total": pass_yds,
            "passing_tds_scored": pass_tds,
            "rushing_yards_total": rush_yds,
            "rushing_tds_scored": rush_tds,
            "receiving_yards_total": rec_yds,
            "receiving_tds_scored": rec_tds,
        })

    return pd.DataFrame(rows)

# -------------------------------------------------------
# 5) fetch ALL players (all 32 teams) – cached
# -------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_all_players(season: int = 2025):
    teams_df = fetch_nfl_teams(season)
    all_rows = []
    for _, row in teams_df.iterrows():
        team_id = row["team_id"]
        team_players = fetch_players_for_team(team_id, season)
        all_rows.append(team_players)
    if all_rows:
        players_df = pd.concat(all_rows, ignore_index=True)
    else:
        players_df = pd.DataFrame(columns=[
            "player","team","position","games_played",
            "passing_yards_total","passing_tds_scored",
            "rushing_yards_total","rushing_tds_scored",
            "receiving_yards_total","receiving_tds_scored"
        ])
    return players_df, teams_df

players_df, teams_df = fetch_all_players()

# -------------------------------------------------------
# 6) fetch team STATS → we will derive DEF tables from this
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_team_stats(season: int = 2025):
    # endpoint name may be /teams/statistics or /games/statistics per team;
    # we’ll try /teams/statistics
    url = f"{BASE_URL}/teams/statistics?league={NFL_LEAGUE_ID}&season={season}"
    r = requests.get(url, headers=HEADERS)
    data = r.json()
    rows = []
    for item in data.get("response", []):
        team = item.get("team", {})
        name = team.get("name", "")
        # offense/defense often under "games" / "passing" / "rushing" / "scoring"
        # we mainly need DEF to mimic your old sheets
        defense = item.get("defense", {})
        # some APIs return "yards" there, we’ll be defensive
        total_yds_allowed = defense.get("yards", 0) or 0
        pass_yds_allowed = defense.get("passing", {}).get("yards", 0) or 0
        rush_yds_allowed = defense.get("rushing", {}).get("yards", 0) or 0
        pass_tds_allowed = defense.get("passing", {}).get("touchdowns", 0) or 0
        rush_tds_allowed = defense.get("rushing", {}).get("touchdowns", 0) or 0
        games_played = item.get("games", {}).get("played", 1) or 1

        rows.append({
            "team": name,
            "games_played": games_played,
            "total_yards_allowed": total_yds_allowed,
            "passing_yards_allowed_total": pass_yds_allowed,
            "rushing_yards_allowed_total": rush_yds_allowed,
            "passing_tds_allowed": pass_tds_allowed,
            "rushing_tds_allowed": rush_tds_allowed,
        })
    return pd.DataFrame(rows)

team_stats_df = fetch_team_stats()

# -------------------------------------------------------
# 7) make FAKE d_rb / d_wr / d_te / d_qb from team_stats_df
# -------------------------------------------------------
# we don’t have per-position allowed like in your Google Sheets,
# so we’ll reuse the *same* table 4 times but map columns so your logic runs
def make_def_tables_from_team_stats(df: pd.DataFrame):
    # QB defense → use passing
    d_qb = df.copy()
    # RB defense → use rushing
    d_rb = df.copy()
    # WR defense → use passing as proxy
    d_wr = df.copy()
    # TE defense → also passing as proxy
    d_te = df.copy()
    return d_rb, d_qb, d_wr, d_te

d_rb, d_qb, d_wr, d_te = make_def_tables_from_team_stats(team_stats_df)

# -------------------------------------------------------
# 8) helper: find player
# -------------------------------------------------------
def find_player_in(df: pd.DataFrame, player_name: str):
    if df is None or df.empty:
        return None
    if "player" not in df.columns:
        return None
    mask = df["player"].astype(str).str.lower() == player_name.lower()
    return df[mask].copy() if mask.any() else None

# detect stat col like your old version
def detect_stat_col(df: pd.DataFrame, prop: str):
    cols = list(df.columns)
    norm = [normalize_header(c) for c in cols]
    mapping = {
        "rushing_yards": ["rushing_yards_total"],
        "receiving_yards": ["receiving_yards_total"],
        "passing_yards": ["passing_yards_total"],
        "receptions": [],  # API-Sports NFL free may not give receptions per player
        "targets": [],
        "carries": ["rushing_attempts_total"],  # may not exist from API
    }
    pri = mapping.get(prop, [])
    for cand in pri:
        if cand in norm:
            return cols[norm.index(cand)]
    return None

def pick_def_df(prop: str, pos: str):
    if prop == "passing_yards":
        return d_qb
    if prop in ["rushing_yards", "carries"]:
        return d_rb
    if prop in ["receiving_yards", "receptions", "targets"]:
        # we don’t have real WR/TE split so just return d_wr
        return d_wr
    return None

def detect_def_col(def_df: pd.DataFrame, prop: str):
    cols = list(def_df.columns)
    norm = [normalize_header(c) for c in cols]
    if prop in ["rushing_yards", "carries"]:
        prefs = ["rushing_yards_allowed_total", "rushing_yards_allowed"]
    elif prop in ["receiving_yards", "receptions", "targets"]:
        prefs = ["passing_yards_allowed_total", "total_yards_allowed"]
    elif prop == "passing_yards":
        prefs = ["passing_yards_allowed_total"]
    else:
        prefs = []
    for cand in prefs:
        if cand in norm:
            return cols[norm.index(cand)]
    # fallback
    for i, nc in enumerate(norm):
        if "allowed" in nc:
            return cols[i]
    return None

# -------------------------------------------------------
# 9) UI (dropdowns)
# -------------------------------------------------------
st.title("🏈 NFL Player Prop Model (API-Sports v8)")

player_list = sorted(players_df["player"].dropna().unique().tolist())
team_list = sorted(teams_df["team"].dropna().unique().tolist())

player_name = st.selectbox("Select Player:", [""] + player_list, index=0)
opponent_team = st.selectbox("Select Opponent Team:", [""] + team_list, index=0)

prop_choices = [
    "passing_yards", "rushing_yards", "receiving_yards",
    "receptions", "targets", "carries", "anytime_td"
]
selected_props = st.multiselect("Select props:", prop_choices, default=["rushing_yards"])

lines = {}
for prop in selected_props:
    if prop != "anytime_td":
        lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=prop)

if not player_name or not opponent_team or not selected_props:
    st.stop()

st.header("📊 Results")

# -------------------------------------------------------
# 10) main loop (props)
# -------------------------------------------------------
for prop in selected_props:
    # --- ANYTIME TD (rush + rec) ---
    if prop == "anytime_td":
        st.subheader("🔥 Anytime TD (Rushing + Receiving + Defense Adjusted)")

        p_row = find_player_in(players_df, player_name)
        if p_row is None or p_row.empty:
            st.warning("Player not found in NFL API data.")
            continue

        gp = float(p_row.iloc[0].get("games_played", 1)) or 1.0

        rec_tds = float(p_row.iloc[0].get("receiving_tds_scored", 0) or 0)
        rush_tds = float(p_row.iloc[0].get("rushing_tds_scored", 0) or 0)
        total_tds = rec_tds + rush_tds

        player_td_pg = total_tds / gp

        # defense side: we only have passing_tds_allowed & rushing_tds_allowed per team
        opp_row = team_stats_df[team_stats_df["team"].astype(str).str.lower() == opponent_team.lower()]

        if not opp_row.empty:
            opp_games = float(opp_row.iloc[0].get("games_played", 1)) or 1.0
            opp_pass_td_pg = safe_div(float(opp_row.iloc[0].get("passing_tds_allowed", 0) or 0), opp_games)
            opp_rush_td_pg = safe_div(float(opp_row.iloc[0].get("rushing_tds_allowed", 0) or 0), opp_games)
            opp_td_pg = opp_pass_td_pg + opp_rush_td_pg
        else:
            # league avg
            opp_td_pg = (team_stats_df["passing_tds_allowed"].fillna(0) +
                         team_stats_df["rushing_tds_allowed"].fillna(0)) / team_stats_df["games_played"].replace(0, np.nan)
            opp_td_pg = opp_td_pg.mean()

        # league TD pg
        league_td_pg = (team_stats_df["passing_tds_allowed"].fillna(0) +
                        team_stats_df["rushing_tds_allowed"].fillna(0)) / team_stats_df["games_played"].replace(0, np.nan)
        league_td_pg = league_td_pg.mean()

        adj_factor = opp_td_pg / league_td_pg if league_td_pg and league_td_pg > 0 else 1.0
        adj_td_rate = player_td_pg * adj_factor
        prob_anytime = float(np.clip(adj_td_rate, 0.0, 1.0))

        st.write(f"**Total TDs (season):** {total_tds:.1f}")
        st.write(f"**Games Played:** {gp:.0f}")
        st.write(f"**Player TDs/Game:** {player_td_pg:.3f}")
        st.write(f"**Opponent TDs/Game (pass+rush):** {opp_td_pg:.3f}")
        st.write(f"**League TDs/Game:** {league_td_pg:.3f}")
        st.write(f"**Adjusted Player TD Rate:** {adj_td_rate:.3f}")
        st.write(f"**Estimated Anytime TD Probability:** {prob_anytime*100:.1f}%")

        bar_df = pd.DataFrame({
            "Category": ["Player TD Rate", "Adj. vs Opponent"],
            "TDs/Game": [player_td_pg, adj_td_rate],
        })
        fig_td = px.bar(bar_df, x="Category", y="TDs/Game",
                        title=f"{player_name} – Anytime TD vs {opponent_team}")
        st.plotly_chart(fig_td, use_container_width=True)
        continue

    # --- OTHER PROPS ---
    p_row = find_player_in(players_df, player_name)
    if p_row is None or p_row.empty:
        st.warning(f"{prop}: player not found.")
        continue

    position = p_row.iloc[0].get("position", "wr")

    stat_col = detect_stat_col(p_row, prop)
    if not stat_col:
        st.warning(f"⚠️ For {prop}, no matching stat column found in NFL data.")
        continue

    season_val = float(p_row.iloc[0].get(stat_col, 0) or 0)
    games_played = float(p_row.iloc[0].get("games_played", 1)) or 1.0
    player_pg = season_val / games_played

    def_df = pick_def_df(prop, position)
    def_col = detect_def_col(def_df, prop) if def_df is not None else None

    opp_allowed_pg = None
    league_allowed_pg = None
    if def_df is not None and def_col is not None:
        if "games_played" in def_df.columns:
            league_allowed_pg = (def_df[def_col] / def_df["games_played"].replace(0, np.nan)).mean()
        else:
            league_allowed_pg = def_df[def_col].mean()
        opp_row = def_df[def_df["team"].astype(str).str.lower() == opponent_team.lower()]
        if not opp_row.empty:
            if "games_played" in opp_row.columns and float(opp_row.iloc[0]["games_played"]) > 0:
                opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played"])
            else:
                opp_allowed_pg = float(opp_row.iloc[0][def_col])
        else:
            opp_allowed_pg = league_allowed_pg

    adj_factor = opp_allowed_pg / league_allowed_pg if league_allowed_pg and league_allowed_pg > 0 else 1.0
    predicted_pg = player_pg * adj_factor

    line_val = lines.get(prop, 0.0)
    stdev = max(3.0, predicted_pg * 0.35)
    z = (line_val - predicted_pg) / stdev
    prob_over = 1 - norm.cdf(z)
    prob_under = norm.cdf(z)
    prob_over = float(np.clip(prob_over, 0.001, 0.999))
    prob_under = float(np.clip(prob_under, 0.001, 0.999))

    st.subheader(prop.replace("_", " ").title())
    st.write(f"**Player (season):** {season_val:.2f} over {games_played:.0f} games → **{player_pg:.2f} per game**")
    st.write(f"**Defense column used:** {def_col}")
    st.write(f"**Adjusted prediction (this game):** {predicted_pg:.2f}")
    st.write(f"**Line:** {line_val}")
    st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
    st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

    fig_bar = px.bar(
        x=["Predicted (this game)", "Line"],
        y=[predicted_pg, line_val],
        title=f"{player_name} – {prop.replace('_', ' ').title()}"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
