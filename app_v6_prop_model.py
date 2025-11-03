# app_v8_props_and_games.py
# Player Props (your working v7.7 core) + NEW "NFL Game Predictor" tab

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Props + Game Predictor (v8)", layout="wide")

# -------------------------------
# 0) URLs
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

# Your NFL scoring sheet with weekly game rows (league-wide).
# (This is your link, converted to CSV export.)
NFL_SCORES_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv&gid=0"

# -------------------------------
# 1) Helpers (shared)
# -------------------------------
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
def load_all_props_data():
    return {name: load_and_clean(url) for name, url in SHEETS.items()}

@st.cache_data(show_spinner=False)
def load_scores_df():
    df = pd.read_csv(NFL_SCORES_URL)
    # Expected headers you told me (normalized):
    df.columns = [normalize_header(c) for c in df.columns]
    # sanity: required columns
    required = [
        "week","date","time","away_team","away_abbr","home_team","home_abbr",
        "away_score","home_score","situation","status","score_text","total_points",
        "game_id","over_under","odds","favored_team","spread","fav_covered",
        "box_score_home","box_score_away","home_display_name","away_display_name",
        "game_winner","game_loser","over_hit","under_hit","broadcast",
        "home_off_yards","away_off_yards"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"⚠️ NFL scoring sheet is missing columns: {missing}")
    # coerce types
    for c in ["week","away_score","home_score","total_points","over_under","spread","home_off_yards","away_off_yards"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -------------------------------
# 2) Player Props (your working v7.7)
# -------------------------------
def make_player_props_tab():
    data = load_all_props_data()
    p_rec, p_rush, p_pass = data["player_receiving"], data["player_rushing"], data["player_passing"]
    d_rb, d_qb, d_wr, d_te = data["def_rb"], data["def_qb"], data["def_wr"], data["def_te"]

    # Sidebar debug
    with st.sidebar:
        st.header("🔎 Debug (Props)")
        st.caption("v7.7 core")
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

    st.subheader("🏈 Player Props")

    # dropdowns
    player_list = sorted(set(
        list(p_rec.get("player", pd.Series(dtype=str)).dropna().unique()) +
        list(p_rush.get("player", pd.Series(dtype=str)).dropna().unique()) +
        list(p_pass.get("player", pd.Series(dtype=str)).dropna().unique())
    ))
    team_list = sorted(set(
        list(d_rb.get("team", pd.Series(dtype=str)).dropna().unique()) +
        list(d_wr.get("team", pd.Series(dtype=str)).dropna().unique()) +
        list(d_te.get("team", pd.Series(dtype=str)).dropna().unique()) +
        list(d_qb.get("team", pd.Series(dtype=str)).dropna().unique())
    ))

    colA, colB = st.columns(2)
    with colA:
        player_name = st.selectbox("Select Player:", options=[""] + player_list, index=0)
    with colB:
        opponent_team = st.selectbox("Select Opponent Team:", options=[""] + team_list, index=0)

    prop_choices = [
        "passing_yards", "rushing_yards", "receiving_yards",
        "receptions", "targets", "carries", "anytime_td"
    ]
    selected_props = st.multiselect("Select props:", prop_choices, default=["receiving_yards"])

    lines = {}
    for prop in selected_props:
        if prop != "anytime_td":
            lines[prop] = st.number_input(f"Sportsbook line for {prop}", value=50.0, key=f"line_{prop}")

    if not player_name or not opponent_team or not selected_props:
        st.info("Pick a player, opponent, and at least one prop.")
        return

    st.header("📊 Prop Results")

    # compute
    for prop in selected_props:
        # Anytime TD combined (rushing + receiving)
        if prop == "anytime_td":
            st.subheader("🔥 Anytime TD (Rushing + Receiving + Defense Adjusted)")

            rec_row = find_player_in(p_rec, player_name)
            rush_row = find_player_in(p_rush, player_name)

            total_tds, total_games = 0.0, 0.0
            for df in [rec_row, rush_row]:
                if df is not None and not df.empty:
                    td_cols = [c for c in df.columns if "td" in c and "allowed" not in c]
                    games_col = "games_played" if "games_played" in df.columns else None
                    if td_cols and games_col:
                        tds = sum(float(df.iloc[0][col]) for col in td_cols if pd.notna(df.iloc[0][col]))
                        total_tds += tds
                        total_games = max(total_games, float(df.iloc[0][games_col]))

            if total_games == 0:
                st.warning("⚠️ No games data found for this player.")
                continue

            player_td_pg = total_tds / total_games

            def_dfs = [d_rb.copy(), d_wr.copy(), d_te.copy()]
            for d in def_dfs:
                if "games_played" not in d.columns:
                    d["games_played"] = 1
                td_like = [c for c in d.columns if "td" in c and "allowed" in c]
                if td_like:
                    d["tds_pg"] = d[td_like].sum(axis=1) / d["games_played"].replace(0, np.nan)
                else:
                    d["tds_pg"] = np.nan

            league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs])
            opp_td_pg = np.nanmean([
                d.loc[d["team"].astype(str).str.lower() == opponent_team.lower(), "tds_pg"].mean()
                for d in def_dfs
            ])
            if np.isnan(opp_td_pg):
                opp_td_pg = league_td_pg

            adj_factor = opp_td_pg / league_td_pg if league_td_pg and league_td_pg > 0 else 1.0
            adj_td_rate = player_td_pg * adj_factor
            prob_anytime = float(np.clip(adj_td_rate, 0.0, 1.0))

            st.write(f"**Total TDs (season):** {total_tds:.1f}")
            st.write(f"**Games Played:** {total_games:.0f}")
            st.write(f"**Player TDs/Game:** {player_td_pg:.2f}")
            st.write(f"**Defense TDs/Game (League Avg):** {league_td_pg:.2f}")
            st.write(f"**Opponent TDs/Game (Adj):** {opp_td_pg:.2f}")
            st.write(f"**Adjusted Player TD Rate:** {adj_td_rate:.2f}")
            st.write(f"**Estimated Anytime TD Probability:** {prob_anytime*100:.1f}%")

            bar_df = pd.DataFrame({
                "Category": ["Player TD Rate", "Adj. vs Opponent"],
                "TDs/Game": [player_td_pg, adj_td_rate],
            })
            fig_td = px.bar(bar_df, x="Category", y="TDs/Game",
                            title=f"{player_name} – Anytime TD vs {opponent_team}")
            st.plotly_chart(fig_td, use_container_width=True)
            continue

        # Other props
        if prop in ["receiving_yards", "receptions", "targets"]:
            player_df = find_player_in(p_rec, player_name); fallback_pos = "wr"
        elif prop in ["rushing_yards", "carries"]:
            player_df = find_player_in(p_rush, player_name); fallback_pos = "rb"
        elif prop == "passing_yards":
            player_df = find_player_in(p_pass, player_name); fallback_pos = "qb"
        else:
            player_df = (find_player_in(p_rec, player_name) or
                         find_player_in(p_rush, player_name) or
                         find_player_in(p_pass, player_name))
            fallback_pos = "wr"

        if player_df is None or player_df.empty:
            st.warning(f"❗ {prop}: player '{player_name}' not found.")
            continue

        player_pos = player_df.iloc[0].get("position", fallback_pos)
        stat_col = detect_stat_col(player_df, prop)

        if not stat_col:
            st.warning(f"⚠️ For {prop}, no matching stat column found.")
            continue

        season_val = float(player_df.iloc[0][stat_col])
        games_played = float(player_df.iloc[0].get("games_played", 1)) or 1.0
        player_pg = season_val / games_played

        def_df = pick_def_df(prop, player_pos)
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
        st.write(f"**Adjusted prediction (this game):** {predicted_pg:.2f}")
        st.write(f"**Line:** {line_val}")
        st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
        st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

        fig_bar = px.bar(x=["Predicted (this game)", "Line"], y=[predicted_pg, line_val],
                         title=f"{player_name} – {prop.replace('_', ' ').title()}")
        st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------
# 3) NFL Game Predictor
# -------------------------------
def make_game_predictor_tab():
    st.subheader("🏟️ NFL Game Predictor")

    df = load_scores_df()
    if df.empty:
        st.warning("Could not load NFL scoring sheet.")
        return

    # Build a long perspective table (team-centric rows)
    # team_name uses display names if present, else abbr.
    home_name = df.get("home_display_name", df.get("home_team", df.get("home_abbr")))
    away_name = df.get("away_display_name", df.get("away_team", df.get("away_abbr")))

    games_home = pd.DataFrame({
        "week": df["week"],
        "team": home_name,
        "opponent": away_name,
        "is_home": True,
        "team_points": df["home_score"],
        "opp_points": df["away_score"]
    })
    games_away = pd.DataFrame({
        "week": df["week"],
        "team": away_name,
        "opponent": home_name,
        "is_home": False,
        "team_points": df["away_score"],
        "opp_points": df["home_score"]
    })
    long_df = pd.concat([games_home, games_away], ignore_index=True)
    # Clean
    for c in ["team","opponent"]:
        long_df[c] = long_df[c].astype(str).str.strip()
    long_df = long_df.dropna(subset=["week","team","opponent"])

    # Controls
    teams = sorted(long_df["team"].dropna().unique().tolist())
    weeks = sorted([int(w) for w in long_df["week"].dropna().unique() if pd.notna(w)])

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        sel_team = st.selectbox("Team", [""] + teams, index=0)
    with col2:
        sel_week = st.selectbox("Week", [""] + [str(w) for w in weeks], index=0)

    if not sel_team or not sel_week:
        st.info("Pick a Team and Week.")
        return

    sel_week = int(sel_week)

    # Find the game for that team/week
    game_row = long_df[(long_df["team"] == sel_team) & (long_df["week"] == sel_week)]
    if game_row.empty:
        st.warning("No game found for that team/week in the sheet.")
        return

    opponent = game_row.iloc[0]["opponent"]
    is_home = bool(game_row.iloc[0]["is_home"])

    # User inputs: spread (team perspective) & O/U
    # spread < 0 means your team is favored; spread > 0 means your team is an underdog
    with col3:
        user_spread = st.number_input("Your Team Spread (team perspective; -1.5 means favored by 1.5)", value= -1.5)
        user_ou = st.number_input("Over/Under", value= 45.5)

    st.write(f"**Opponent:** {opponent} {'(Home)' if is_home else '(Away)'}")

    # Historic stats up to prior weeks
    prior_team = long_df[(long_df["team"] == sel_team) & (long_df["week"] < sel_week)]
    prior_opp  = long_df[(long_df["team"] == opponent) & (long_df["week"] < sel_week)]

    # Fallback: if no prior, include all weeks
    if prior_team.empty:
        prior_team = long_df[(long_df["team"] == sel_team)]
    if prior_opp.empty:
        prior_opp = long_df[(long_df["team"] == opponent)]

    # Compute offense/defense averages
    team_off_ppg = prior_team["team_points"].mean()
    team_def_papg = prior_team["opp_points"].mean()
    opp_off_ppg = prior_opp["team_points"].mean()
    opp_def_papg = prior_opp["opp_points"].mean()

    # league averages as guardrails
    league_off_ppg = long_df["team_points"].mean()
    league_def_papg = long_df["opp_points"].mean()

    # blend offense with opponent defense (simple mean), guardrail with league if NaNs
    def nz(x, fallback):
        return fallback if pd.isna(x) else x

    team_off_ppg = nz(team_off_ppg, league_off_ppg)
    team_def_papg = nz(team_def_papg, league_def_papg)
    opp_off_ppg  = nz(opp_off_ppg, league_off_ppg)
    opp_def_papg = nz(opp_def_papg, league_def_papg)

    exp_team = (team_off_ppg + opp_def_papg) / 2.0
    exp_opp  = (opp_off_ppg  + team_def_papg) / 2.0

    # Home-field tweak (~ +1.5 points)
    if is_home:
        exp_team += 1.5
    else:
        exp_opp += 1.5

    exp_total = exp_team + exp_opp
    exp_margin = exp_team - exp_opp  # positive means your team by N points

    # Probabilities using normal approximation
    # Stdev assumptions (tunable):
    stdev_margin = 13.5
    stdev_total  = 10.0

    # Cover probability: margin - spread > 0
    z_cover = (exp_margin - user_spread) / stdev_margin
    prob_cover = float(1 - norm.cdf( -z_cover ))  # P(margin - spread > 0)

    # Win probability: margin > 0
    z_win = (exp_margin - 0.0) / stdev_margin
    prob_win = float(1 - norm.cdf( -z_win ))

    # Over/Under
    z_over = (exp_total - user_ou) / stdev_total
    prob_over = float(1 - norm.cdf( -z_over ))
    prob_under = 1 - prob_over

    # Display
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📈 Model Projection")
        st.write(f"**Expected Score:** {sel_team} {exp_team:.1f} – {opponent} {exp_opp:.1f}")
        st.write(f"**Expected Margin ({sel_team}):** {exp_margin:+.1f}")
        st.write(f"**Expected Total:** {exp_total:.1f}")
        st.write(f"**Win Probability ({sel_team}):** {prob_win*100:.1f}%")
        st.write(f"**Cover Probability (spread {user_spread:+.1f}):** {prob_cover*100:.1f}%")
        st.write(f"**Over {user_ou:.1f}:** {prob_over*100:.1f}%  |  **Under:** {prob_under*100:.1f}%")

    with c2:
        bars = pd.DataFrame({
            "Metric": ["Expected Points - Your Team", "Expected Points - Opponent", "Expected Total"],
            "Value": [exp_team, exp_opp, exp_total]
        })
        fig = px.bar(bars, x="Metric", y="Value",
                     title=f"{sel_team} vs {opponent} — Week {sel_week} Projection")
        st.plotly_chart(fig, use_container_width=True)

    # Quick league-wide context this week (overs vs unders)
    this_week = df[df["week"] == sel_week]
    if not this_week.empty and "over_hit" in this_week.columns and "under_hit" in this_week.columns:
        over_cnt = int((this_week["over_hit"] == True).sum())
        under_cnt = int((this_week["under_hit"] == True).sum())
        st.markdown("### 📊 League context (this week)")
        fig2 = px.bar(
            pd.DataFrame({"Outcome":["Over","Under"],"Games":[over_cnt, under_cnt]}),
            x="Outcome", y="Games",
            title=f"Week {sel_week}: Overs vs Unders (from your sheet)"
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# 4) Layout – Tabs
# -------------------------------
tab1, tab2 = st.tabs(["🎯 Player Props", "🧠 NFL Game Predictor"])
with tab1:
    make_player_props_tab()
with tab2:
    make_game_predictor_tab()
