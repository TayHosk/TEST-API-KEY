import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Prop & Game Model", layout="wide")

# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.radio(
    "Select Page:",
    ["🏈 Player Prop Model", "📈 NFL Game Predictor"]
)
st.sidebar.markdown("---")
st.sidebar.caption("NFL Data Model – Combined v8.7")

# ======================================================
# 🏈 TAB 1: PLAYER PROP MODEL (v7.7)
# ======================================================
if page == "🏈 Player Prop Model":
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
        name = str(name).strip().replace(" ", "_").lower()
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

    st.header("📊 Results")

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

    for prop in selected_props:
        if prop == "anytime_td":
            st.subheader("🔥 Anytime TD (Rushing + Receiving + Defense Adjusted)")
            rec_row = find_player_in(p_rec, player_name)
            rush_row = find_player_in(p_rush, player_name)
            total_tds = 0.0
            total_games = 0.0
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
                d["tds_pg"] = (
                    d[[c for c in d.columns if "td" in c and "allowed" in c]].sum(axis=1)
                    / d["games_played"].replace(0, np.nan)
                )
            league_td_pg = np.nanmean([d["tds_pg"].mean() for d in def_dfs])
            opp_td_pg = np.nanmean([
                d.loc[d["team"].astype(str).str.lower() == opponent_team.lower(), "tds_pg"].mean()
                for d in def_dfs
            ])
            if np.isnan(opp_td_pg):
                opp_td_pg = league_td_pg
            adj_factor = opp_td_pg / league_td_pg if league_td_pg > 0 else 1.0
            adj_td_rate = player_td_pg * adj_factor
            prob_anytime = min(adj_td_rate, 1.0)
            st.write(f"**Estimated Anytime TD Probability:** {prob_anytime*100:.1f}%")
            continue

        # --- Other Props ---
        if prop in ["receiving_yards", "receptions", "targets"]:
            player_df = find_player_in(p_rec, player_name)
            fallback_pos = "wr"
        elif prop in ["rushing_yards", "carries"]:
            player_df = find_player_in(p_rush, player_name)
            fallback_pos = "rb"
        elif prop == "passing_yards":
            player_df = find_player_in(p_pass, player_name)
            fallback_pos = "qb"
        else:
            player_df = find_player_in(p_rec, player_name) or find_player_in(p_rush, player_name) or find_player_in(p_pass, player_name)
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
        st.write(f"**Adjusted prediction (this game):** {predicted_pg:.2f}")
        st.write(f"**Line:** {line_val}")
        st.write(f"**Probability of OVER:** {prob_over*100:.1f}%")
        st.write(f"**Probability of UNDER:** {prob_under*100:.1f}%")

        fig_bar = px.bar(x=["Predicted (this game)", "Line"], y=[predicted_pg, line_val],
                         title=f"{player_name} – {prop.replace('_', ' ').title()}")
        st.plotly_chart(fig_bar, use_container_width=True)

# ======================================================
# 📈 TAB 2: NFL GAME PREDICTOR (Vegas-Calibrated)
# ======================================================
elif page == "📈 NFL Game Predictor":
    st.title("📈 NFL Game Predictor (Vegas-Calibrated)")

    SCORE_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"

    @st.cache_data(show_spinner=False)
    def load_scores():
        df = pd.read_csv(SCORE_URL)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    scores_df = load_scores()
    if scores_df.empty:
        st.error("❌ Could not load NFL data.")
        st.stop()

    week_list = sorted(scores_df["week"].dropna().unique())
    team_list = sorted(set(scores_df["home_team"].dropna().unique()) | set(scores_df["away_team"].dropna().unique()))

    st.subheader("Select Game Details")
    selected_week = st.selectbox("Select NFL Week:", week_list)
    selected_team = st.selectbox("Select Your Team:", team_list)

    game = scores_df[
        ((scores_df["home_team"] == selected_team) | (scores_df["away_team"] == selected_team))
        & (scores_df["week"] == selected_week)
    ]

    if game.empty:
        st.warning("No game data found for that team/week.")
    else:
        g = game.iloc[0]
        opponent = g["away_team"] if g["home_team"] == selected_team else g["home_team"]

        st.write(f"**Matchup:** {selected_team} vs {opponent}")
        over_under = st.number_input("Enter Over/Under line:", value=float(g.get("over_under", 45.0)))
        spread = st.number_input("Enter Spread (negative = favorite):", value=float(g.get("spread", 0.0)))

        def avg_scoring(df, team):
            scored_home = df.loc[df["home_team"] == team, "home_score"].mean()
            scored_away = df.loc[df["away_team"] == team, "away_score"].mean()
            allowed_home = df.loc[df["home_team"] == team, "away_score"].mean()
            allowed_away = df.loc[df["away_team"] == team, "home_score"].mean()
            return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

        team_avg_scored, team_avg_allowed = avg_scoring(scores_df, selected_team)
        opp_avg_scored, opp_avg_allowed = avg_scoring(scores_df, opponent)

        raw_team_pts = (team_avg_scored + opp_avg_allowed) / 2
        raw_opp_pts = (opp_avg_scored + team_avg_allowed) / 2

        league_avg_pts = scores_df[["home_score", "away_score"]].stack().mean()
        cal_factor = 22.3 / league_avg_pts if not np.isnan(league_avg_pts) and league_avg_pts > 0 else 1.0
        raw_team_pts *= cal_factor
        raw_opp_pts *= cal_factor

        total_pred = raw_team_pts + raw_opp_pts
        margin = raw_team_pts - raw_opp_pts

        total_diff = total_pred - over_under
        spread_diff = margin - (-spread)

        st.markdown(f"""
        ### 🧮 Vegas-Calibrated Projection
        **Predicted Score:**  
        {selected_team}: **{raw_team_pts:.1f}**  
        {opponent}: **{raw_opp_pts:.1f}**

        **Predicted Total:** {total_pred:.1f}  
        **Vegas O/U:** {over_under:.1f}  
        **→ Lean:** {"Over" if total_diff > 0 else "Under"} ({abs(total_diff):.1f} pts)

        **Spread Line:** {spread:+.1f}  
        **→ Lean:** {selected_team if spread_diff > 0 else opponent} to cover ({abs(spread_diff):.1f} pts)
        """)

        # --- Charts ---
        fig_total = px.bar(
            x=["Predicted Total", "Vegas Line"],
            y=[total_pred, over_under],
            title="Predicted Total vs Vegas Line"
        )
        st.plotly_chart(fig_total, use_container_width=True)

        fig_margin = px.bar(
            x=["Predicted Margin", "Vegas Spread"],
            y=[margin, -spread],
            title="Predicted Margin vs Vegas Spread"
        )
        st.plotly_chart(fig_margin, use_container_width=True)
