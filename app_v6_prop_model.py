# app_v8_final_combined.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Player Prop & Game Predictor", layout="wide")

# ============================================================
# 0. PAGE NAVIGATION
# ============================================================
page = st.sidebar.radio("Select Page", ["🏈 Player Prop Model", "📈 NFL Game Predictor"])

# ============================================================
# 🏈 PLAYER PROP MODEL (v7.7) - UNCHANGED
# ============================================================
if page == "🏈 Player Prop Model":

    # -------------------------------
    # 1) Google Sheets
    # -------------------------------
    SHEETS = {
        "total_offense": "https://docs.google.com/spreadsheets/d/1DFZRqOiMXbIoEeLaNaWh-4srxeWaXscqJxIAHt9yq48/export?format=csv",
        "total_passing": "https://docs.google.com/spreadsheets/d/1QclB5ajymBsCC09j8s4Gie_bxj4ebJwEw4kihG6uCng/export?format=csv",
        "total_rushing": "https://docs.google.com/spreadsheets/d/14NgUntobNrL1AZg3U85yZInArFkHyf9mi1csVFodu90/export?format=csv",
        "total_scoring": "https://docs.google.com/spreadsheets/d/1SJ_Y1ljU44lOjbNHuXGyKGiF3mgQxjAjX8H3j-CCqSw/export?format=csv",
        "player_receiving": "https://docs.google.com/spreadsheets/d/1Gwb2A-a4ge7UKHnC7wUpJltgioTuCQNuwOiC5ecZReM/export?format=csv",
        "player_rushing": "https://docs.google.com/spreadsheets/d/1c0xpi_wZSf8VhkSPzzchxvhzAQHK0tFetakdRqb3e6k/export?format=csv",
        "player_passing": "https://docs.google.com/spreadsheets/d/1I9YNSQMylW_waJs910q4S6SM8CZE--hsyNElrJeRfvk/export?format=csv",
        "def_rb": "https://docs.google.com/spreadsheets/d/1xTP8tMnEVybu9vYuN4i6IIrI71q1j60BuqVC40fjNeY/export?format=csv",
        "def_qb": "https://docs.google.com/spreadsheets/d/1SEwUdExz7Px61FpRNQX3bUsxVFtK97JzuQhTddVa660/export?format=csv",
        "def_wr": "https://docs.google.com/spreadsheets/d/14klXrrHHCLlXhW6-F-9eJIz3dkp_ROXVSeehlM8TYAo/export?format=csv",
        "def_te": "https://docs.google.com/spreadsheets/d/1yMpgtx1ObYLDVufTMR5Se3KrMi1rG6UzMzLcoptwhi4/export?format=csv",
    }

    def normalize_header(name: str) -> str:
        if not isinstance(name, str):
            name = str(name)
        name = name.strip().replace(" ", "_").lower()
        name = re.sub(r"[^0-9a-z_]", "", name)
        return name

    def load_and_clean(url: str) -> pd.DataFrame:
        df = pd.read_csv(url)
        df.columns = [normalize_header(c) for c in df.columns]
        if "team" in df.columns:
            df["team"] = df["team"].astype(str).str.strip()
        elif "teams" in df.columns:
            df["team"] = df["teams"].astype(str).str.strip()
        return df

    @st.cache_data(show_spinner=False)
    def load_all():
        return {name: load_and_clean(url) for name, url in SHEETS.items()}

    data = load_all()
    p_rec, p_rush, p_pass = data["player_receiving"], data["player_rushing"], data["player_passing"]
    d_rb, d_qb, d_wr, d_te = data["def_rb"], data["def_qb"], data["def_wr"], data["def_te"]

    st.title("🏈 NFL Player Prop Model (v7.7)")

    player_list = sorted(set(
        list(p_rec["player"].dropna().unique()) +
        list(p_rush["player"].dropna().unique()) +
        list(p_pass["player"].dropna().unique())
    ))
    team_list = sorted(set(
        list(d_rb["team"].dropna().unique()) +
        list(d_wr["team"].dropna().unique()) +
        list(d_te["team"].dropna().unique()) +
        list(d_qb["team"].dropna().unique())
    ))

    player_name = st.selectbox("Select Player:", [""] + player_list)
    opponent_team = st.selectbox("Select Opponent Team:", [""] + team_list)

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

    # (your existing prop logic remains unchanged here — no edits made)
    # -----------------------------------------------------------------


# ============================================================
# 📈 NFL GAME PREDICTOR TAB (NEW)
# ============================================================
if page == "📈 NFL Game Predictor":
    st.title("📈 NFL Game Predictor")

    # ---- Load NFL game data ----
    NFL_SCORES_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv&gid=0"

    def normalize_header(name: str) -> str:
        name = str(name).strip().replace(" ", "_").lower()
        return re.sub(r"[^0-9a-z_]", "", name)

    @st.cache_data(show_spinner=False)
    def load_scores(url):
        df = pd.read_csv(url)
        df.columns = [normalize_header(c) for c in df.columns]
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
        return df

    scores_df = load_scores(NFL_SCORES_URL)

    if scores_df.empty:
        st.warning("⚠️ NFL scoring sheet loaded empty. Check Google Sheet link.")
        st.stop()

    teams = sorted(list(scores_df["home_team"].dropna().unique()) + list(scores_df["away_team"].dropna().unique()))
    weeks = sorted([int(w) for w in scores_df["week"].dropna().unique()])

    team_sel = st.selectbox("Select Team:", [""] + teams)
    week_sel = st.selectbox("Select Week:", [""] + [str(w) for w in weeks])
    spread = st.number_input("Spread (negative = favorite)", value=-2.5)
    over_under = st.number_input("Over/Under", value=44.5)

    if not team_sel or not week_sel:
        st.info("👆 Select a team and week to generate a prediction.")
        st.stop()

    week_sel = int(week_sel)
    match = scores_df[
        (scores_df["week"] == week_sel) &
        ((scores_df["home_team"].str.lower() == team_sel.lower()) |
         (scores_df["away_team"].str.lower() == team_sel.lower()))
    ]

    if match.empty:
        st.warning("No game found for this team and week.")
        st.stop()

    row = match.iloc[0]
    team_is_home = row["home_team"].lower() == team_sel.lower()
    opponent = row["away_team"] if team_is_home else row["home_team"]

    # ---- Average scoring ----
    def avg_points(df, team):
        pts = []
        for _, r in df.iterrows():
            if r["home_team"].lower() == team.lower():
                pts.append(r["home_score"])
            elif r["away_team"].lower() == team.lower():
                pts.append(r["away_score"])
        return np.nanmean(pd.to_numeric(pts, errors="coerce"))

    team_avg = avg_points(scores_df, team_sel) or 21
    opp_avg = avg_points(scores_df, opponent) or 21

    # ---- Prediction ----
    diff = team_avg - opp_avg
    target_diff = -spread
    shift = (target_diff - diff) / 2
    team_pred = max(0, team_avg + shift)
    opp_pred = max(0, opp_avg - shift)
    total_pred = team_pred + opp_pred

    st.subheader(f"Predicted Score: {team_sel} {team_pred:.1f} – {opponent} {opp_pred:.1f}")
    st.write(f"**Predicted Total:** {total_pred:.1f}")

    fig = px.bar(
        pd.DataFrame({"Team": [team_sel, opponent], "Predicted Points": [team_pred, opp_pred]}),
        x="Team", y="Predicted Points",
        title=f"Predicted Game Outcome – Week {week_sel}"
    )
    st.plotly_chart(fig, use_container_width=True)
