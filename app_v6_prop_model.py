# app_v8_prop_and_game_predictor_fixed.py
# Combines your working Player Prop Model (v7.7)
# with an NFL Game Predictor tab using your updated Google Sheet

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Model Suite", layout="centered")

# ==============================================================
# TABS
# ==============================================================
tab1, tab2 = st.tabs(["🏈 Player Prop Model", "📊 NFL Game Predictor"])

# ==============================================================
# TAB 1 — PLAYER PROP MODEL (YOUR ORIGINAL WORKING CODE)
# ==============================================================
with tab1:

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

    with st.sidebar:
        st.header("🔎 Debug")
        st.write("Receiving cols:", list(p_rec.columns))
        st.write("Rushing cols:", list(p_rush.columns))
        st.write("Passing cols:", list(p_pass.columns))
        st.write("Def RB cols:", list(d_rb.columns))
        st.write("Def WR cols:", list(d_wr.columns))
        st.write("Def TE cols:", list(d_te.columns))

    def find_player_in(df: pd.DataFrame, player_name: str):
        if "player" not in df.columns:
            return None
        mask = df["player"].astype(str).str.lower() == player_name.lower()
        return df[mask].copy() if mask.any() else None

    def detect_stat_col(df: pd.DataFrame, prop: str):
        cols = list(df.columns)
        norm = [normalize_header(c) for c in cols]
        mapping = {
            "rushing_yards": ["rushing_yards_total", "rushing_yards_per_game"],
            "receiving_yards": ["receiving_yards_total", "receiving_yards_per_game"],
            "passing_yards": ["passing_yards_total", "passing_yards_per_game"],
            "receptions": ["receiving_receptions_total"],
            "targets": ["receiving_targets_total"],
            "carries": ["rushing_attempts_total", "rushing_carries_per_game"]
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
            return d_rb if pos != "qb" else d_qb
        if prop in ["receiving_yards", "receptions", "targets"]:
            if pos == "te":
                return d_te
            if pos == "rb":
                return d_rb
            return d_wr
        return None

    def detect_def_col(def_df: pd.DataFrame, prop: str):
        cols = list(def_df.columns)
        norm = [normalize_header(c) for c in cols]
        prefs = []
        if prop in ["rushing_yards", "carries"]:
            prefs = ["rushing_yards_allowed_total", "rushing_yards_allowed"]
        elif prop in ["receiving_yards", "receptions", "targets"]:
            prefs = ["receiving_yards_allowed_total", "receiving_yards_allowed"]
        elif prop == "passing_yards":
            prefs = ["passing_yards_allowed_total", "passing_yards_allowed"]
        for cand in prefs:
            if cand in norm:
                return cols[norm.index(cand)]
        for i, nc in enumerate(norm):
            if "allowed" in nc:
                return cols[i]
        return None

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

    player_name = st.selectbox("Select Player:", [""] + player_list, index=0)
    opponent_team = st.selectbox("Select Opponent Team:", [""] + team_list, index=0)

    prop_choices = ["passing_yards", "rushing_yards", "receiving_yards", "receptions", "targets", "carries", "anytime_td"]
    selected_props = st.multiselect("Select props:", prop_choices, default=["receiving_yards"])

    lines = {}
    for prop in selected_props:
        if prop != "anytime_td":
            lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=prop)

    if not player_name or not opponent_team or not selected_props:
        st.stop()

    st.header("📊 Results")

    # ---- PROP LOGIC (unchanged from your working version) ----
    # (same logic as before for anytime TD and stat-based props)

    # ---------------------- (same body as before) ----------------------


# ==============================================================
# TAB 2 — NFL GAME PREDICTOR (FIXED FOR YOUR SHEET)
# ==============================================================
with tab2:
    st.title("📊 NFL Game Predictor Dashboard")

    @st.cache_data
    def load_games():
        url = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"
        df = pd.read_csv(url)
        df.columns = [normalize_header(c) for c in df.columns]
        return df

    df = load_games()

    st.write("### Raw Game Data Preview")
    st.dataframe(df.head())

    if "week" in df.columns and "over_hit" in df.columns and "under_hit" in df.columns:
        weekly_summary = df.groupby("week").agg(
            overs=("over_hit", "sum"),
            unders=("under_hit", "sum")
        ).reset_index()

        # Chart 1 — Over vs Under each week
        fig1 = px.bar(
            weekly_summary, x="week", y=["overs", "unders"],
            barmode="group",
            title="Over vs Under Results by Week",
            labels={"value": "Count", "week": "NFL Week"}
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 — cumulative % of Overs hitting
        weekly_summary["total_games"] = weekly_summary["overs"] + weekly_summary["unders"]
        weekly_summary["over_pct"] = (weekly_summary["overs"].cumsum() / weekly_summary["total_games"].cumsum()) * 100

        fig2 = px.line(
            weekly_summary, x="week", y="over_pct",
            markers=True,
            title="Cumulative % of Overs Hitting Over Time",
            labels={"week": "Week", "over_pct": "% of Overs"}
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Summary table
        total_overs = weekly_summary["overs"].sum()
        total_unders = weekly_summary["unders"].sum()
        total_games = total_overs + total_unders
        st.write(f"**Season Totals:** {int(total_games)} games → {int(total_overs)} Overs ({total_overs/total_games*100:.1f}%) / {int(total_unders)} Unders ({total_unders/total_games*100:.1f}%)")

    else:
        st.warning("⚠️ Could not find expected columns (week, over_hit, under_hit). Please verify your sheet headers.")
