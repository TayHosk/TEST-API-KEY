import streamlit as st
import pandas as pd
import numpy as np
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
st.sidebar.caption("Biosense NFL Data Model – v8.1 (Vegas-Calibrated)")

# ======================================================
# 🏈 TAB 1: PLAYER PROP MODEL (Your working version)
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
    st.title("🏈 NFL Player Prop Model (v7.7)")
    st.success("✅ Player Prop Model working correctly.")

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

    # --- Dropdown Inputs ---
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

        # --- Team averages ---
        def avg_scoring(df, team):
            scored_home = df.loc[df["home_team"] == team, "home_score"].mean()
            scored_away = df.loc[df["away_team"] == team, "away_score"].mean()
            allowed_home = df.loc[df["home_team"] == team, "away_score"].mean()
            allowed_away = df.loc[df["away_team"] == team, "home_score"].mean()
            return np.nanmean([scored_home, scored_away]), np.nanmean([allowed_home, allowed_away])

        team_avg_scored, team_avg_allowed = avg_scoring(scores_df, selected_team)
        opp_avg_scored, opp_avg_allowed = avg_scoring(scores_df, opponent)

        # --- Raw projection ---
        raw_team_pts = (team_avg_scored + opp_avg_allowed) / 2
        raw_opp_pts = (opp_avg_scored + team_avg_allowed) / 2

        # --- Vegas calibration: Adjust league average to ~22.3 PPG per team ---
        league_avg_pts = scores_df[["home_score", "away_score"]].stack().mean()
        cal_factor = 22.3 / league_avg_pts if not np.isnan(league_avg_pts) and league_avg_pts > 0 else 1.0
        raw_team_pts *= cal_factor
        raw_opp_pts *= cal_factor

        total_pred = raw_team_pts + raw_opp_pts
        margin = raw_team_pts - raw_opp_pts

        # --- Compare to Vegas lines ---
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
