# app_final.py
# ============================================
# NFL Player Prop Model (your working code) + NFL Game Predictor (new tab)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Player Props & Game Predictor", layout="wide")

# ============================================================
# PAGE NAVIGATION  — must be at the very top, before any UI
# ============================================================
page = st.sidebar.radio("Select Page", ["🏈 Player Prop Model", "📈 NFL Game Predictor"])

# -------------------------------
# Shared helpers
# -------------------------------
def normalize_header(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"[^0-9a-z_]", "", name)
    return name

def _read_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [normalize_header(c) for c in df.columns]
    return df

# ============================================================
# 🏈 PAGE 1: Player Prop Model (your working v7.7)
# ============================================================
if page == "🏈 Player Prop Model":
    # -------------------------------
    # 1) Google Sheets (your original)
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

    @st.cache_data(show_spinner=False)
    def load_and_clean(url: str) -> pd.DataFrame:
        df = _read_csv(url)
        # normalize team
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

    # -------------------------------
    # Sidebar Debug (your original)
    # -------------------------------
    with st.sidebar:
        st.header("🔎 Debug – Props")
        st.write("Receiving cols:", list(p_rec.columns))
        st.write("Rushing cols:", list(p_rush.columns))
        st.write("Passing cols:", list(p_pass.columns))
        st.write("Def RB cols:", list(d_rb.columns))
        st.write("Def WR cols:", list(d_wr.columns))
        st.write("Def TE cols:", list(d_te.columns))
        st.write("This page matches your working v7.7 logic.")

    # -------------------------------
    # Helper Functions (your original)
    # -------------------------------
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

    # -------------------------------
    # UI Dropdowns (your original)
    # -------------------------------
    st.title("🏈 NFL Player Prop Model (v7.7)")

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

    player_name = st.selectbox(
        "Select Player:",
        options=[""] + player_list,
        index=0,
        help="Start typing a player's name to search"
    )
    opponent_team = st.selectbox(
        "Select Opponent Team:",
        options=[""] + team_list,
        index=0,
        help="Start typing a team's name to search"
    )

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
        st.stop()

    st.header("📊 Results")

    # -------------------------------
    # Prop Logic (your working v7.7)
    # -------------------------------
    for prop in selected_props:
        # --- ANYTIME TD (Rushing + Receiving Combined) ---
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

            # defensive TD context
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
                d.loc[d.get("team", pd.Series(dtype=str)).astype(str).str.lower() == opponent_team.lower(), "tds_pg"].mean()
                for d in def_dfs
            ])
            if np.isnan(opp_td_pg):
                opp_td_pg = league_td_pg

            adj_factor = opp_td_pg / league_td_pg if league_td_pg > 0 else 1.0
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
                            title=f"{player_name} – Anytime TD (Rushing + Receiving) vs {opponent_team}")
            st.plotly_chart(fig_td, use_container_width=True)
            continue

        # --- OTHER PROPS
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

            opp_row = def_df[def_df.get("team", pd.Series(dtype=str)).astype(str).str.lower() == opponent_team.lower()]
            if not opp_row.empty:
                if "games_played" in opp_row.columns and float(opp_row.iloc[0]["games_played"]) > 0:
                    opp_allowed_pg = float(opp_row.iloc[0][def_col]) / float(opp_row.iloc[0]["games_played"])
                else:
                    opp_allowed_pg = float(opp_row.iloc[0][def_col])
            else:
                opp_allowed_pg = league_allowed_pg

        adj_factor = opp_allowed_pg / league_allowed_pg if (league_allowed_pg is not None and league_allowed_pg > 0 and opp_allowed_pg is not None) else 1.0
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


# ============================================================
# 📈 PAGE 2: NFL Game Predictor (new, with working dropdowns)
# ============================================================
if page == "📈 NFL Game Predictor":
    st.title("📈 NFL Game Predictor")

    # Your sheet share link → convert to CSV export:
    # https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/edit?gid=0#gid=0
    SCORES_CSV = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv&gid=0"

    @st.cache_data(show_spinner=False)
    def load_scores_df() -> pd.DataFrame:
        df = _read_csv(SCORES_CSV)
        # expected columns (normalized):
        # week, date, time, away_team, away_abbr, home_team, home_abbr, away_score, home_score,
        # situation, status, score_text, total_points, game_id, over_under, odds, favored_team, spread,
        # fav_covered, box_score_home, box_score_away, home_display_name, away_display_name, game_winner,
        # game_loser, over_hit, under_hit, broadcast, home_off_yards, away_off_yards
        return df

    scores = load_scores_df()

    # Show a tiny debug if dropdowns look empty
    with st.expander("🔎 Debug – Scores sheet (click to view)"):
        st.write("Columns:", list(scores.columns))
        st.dataframe(scores.head(10))

    # Build Week dropdown
    week_col = "week" if "week" in scores.columns else None
    if week_col is None:
        st.error("Could not find a 'week' column in your NFL scores sheet.")
        st.stop()

    weeks = sorted([w for w in scores[week_col].dropna().unique().tolist() if str(w).strip() != ""])
    weeks = [int(w) if str(w).isdigit() else w for w in weeks]

    # Build Team dropdown from home_team & away_team (fallback to display names if needed)
    team_candidates = []
    if "home_team" in scores.columns:
        team_candidates += scores["home_team"].dropna().astype(str).tolist()
    if "away_team" in scores.columns:
        team_candidates += scores["away_team"].dropna().astype(str).tolist()

    # fallback if those are missing or empty — try display names
    if len(team_candidates) == 0:
        if "home_display_name" in scores.columns:
            team_candidates += scores["home_display_name"].dropna().astype(str).tolist()
        if "away_display_name" in scores.columns:
            team_candidates += scores["away_display_name"].dropna().astype(str).tolist()

    teams = sorted(set([t for t in team_candidates if t.strip() != ""]))

    # UI selectors
    colA, colB = st.columns([1, 2], gap="large")
    with colA:
        sel_week = st.selectbox("Select Week", options=[""] + weeks, index=0)
        sel_team = st.selectbox("Select Team", options=[""] + teams, index=0, help="Start typing to search")

    if not sel_week or not sel_team:
        st.info("Choose a **Week** and **Team** to see that matchup and add your lines.")
        st.stop()

    # Locate the game row for that week/team (home or away)
    is_this_week = scores[week_col].astype(str) == str(sel_week)
    is_home = scores.get("home_team", pd.Series("", index=scores.index)).astype(str).str.lower() == sel_team.lower()
    is_away = scores.get("away_team", pd.Series("", index=scores.index)).astype(str).str.lower() == sel_team.lower()

    game = scores[is_this_week & (is_home | is_away)].copy()

    if game.empty:
        st.warning("No game found for that Week/Team in the sheet. Double-check team spelling matches the sheet’s values.")
        st.stop()

    # if multiple rows (rare), take the first
    game = game.iloc[0]

    # Identify opponent
    if "home_team" in scores.columns and "away_team" in scores.columns:
        if str(game["home_team"]).lower() == sel_team.lower():
            opponent = str(game["away_team"])
            sel_side = "home"
        else:
            opponent = str(game["home_team"])
            sel_side = "away"
    else:
        # fallback using display name if needed
        if "home_display_name" in scores.columns and "away_display_name" in scores.columns:
            if str(game["home_display_name"]).lower() == sel_team.lower():
                opponent = str(game["away_display_name"])
                sel_side = "home"
            else:
                opponent = str(game["home_display_name"])
                sel_side = "away"
        else:
            opponent = "Unknown"
            sel_side = "home"

    st.subheader(f"Week {sel_week}: {sel_team} vs {opponent}")

    # Pre-fill lines if present
    over_under_val = None
    if "over_under" in scores.index or "over_under" in scores.columns:
        try:
            over_under_val = float(game.get("over_under", np.nan))
        except Exception:
            over_under_val = None

    spread_val = None
    spread_raw = None
    favored = None
    if "spread" in scores.index or "spread" in scores.columns:
        try:
            spread_raw = float(game.get("spread", np.nan))  # positive always attached to favored team in sheet
        except Exception:
            spread_raw = None
    if "favored_team" in scores.index or "favored_team" in scores.columns:
        favored = str(game.get("favored_team", "")).strip()

    # Convert sheet favored spread into team-relative spread for selected team
    # If selected team is favored, show negative; otherwise positive.
    if spread_raw is not None and favored:
        if favored.lower() == sel_team.lower():
            spread_val = -abs(spread_raw)
        elif favored.lower() == opponent.lower():
            spread_val = abs(spread_raw)
        else:
            spread_val = spread_raw

    with colB:
        user_ou = st.number_input("Your Over/Under line (total points)", value=float(over_under_val) if over_under_val else 44.5, step=0.5)
        user_spread = st.number_input(
            f"{sel_team} spread (negative = favored, positive = underdog)",
            value=float(spread_val) if spread_val is not None else 0.0, step=0.5
        )

    st.markdown("—")

    # ======== A very simple baseline model using historical scoring in the sheet =========
    # Compute team average scored & allowed from the sheet (all rows available)
    def team_points_scored(df: pd.DataFrame, team: str) -> pd.Series:
        hs = df.get("home_score", pd.Series(dtype=float))
        as_ = df.get("away_score", pd.Series(dtype=float))
        ht = df.get("home_team", pd.Series(dtype=str)).astype(str)
        at = df.get("away_team", pd.Series(dtype=str)).astype(str)
        scored = pd.Series(0.0, index=df.index)
        scored = np.where(ht.str.lower() == team.lower(), hs, scored)
        scored = np.where(at.str.lower() == team.lower(), as_, scored)
        return pd.to_numeric(scored, errors="coerce")

    def team_points_allowed(df: pd.DataFrame, team: str) -> pd.Series:
        hs = df.get("home_score", pd.Series(dtype=float))
        as_ = df.get("away_score", pd.Series(dtype=float))
        ht = df.get("home_team", pd.Series(dtype=str)).astype(str)
        at = df.get("away_team", pd.Series(dtype=str)).astype(str)
        allowed = pd.Series(0.0, index=df.index)
        # if team is home in a row, it allowed away_score; if team is away, it allowed home_score
        allowed = np.where(ht.str.lower() == team.lower(), as_, allowed)
        allowed = np.where(at.str.lower() == team.lower(), hs, allowed)
        return pd.to_numeric(allowed, errors="coerce")

    # Use only rows with numeric scores
    valid_rows = scores.copy()
    for c in ["home_score", "away_score"]:
        if c not in valid_rows.columns:
            valid_rows[c] = np.nan
    valid_rows["home_score"] = pd.to_numeric(valid_rows["home_score"], errors="coerce")
    valid_rows["away_score"] = pd.to_numeric(valid_rows["away_score"], errors="coerce")
    valid_rows = valid_rows.dropna(subset=["home_score", "away_score"], how="any")

    if valid_rows.empty:
        st.warning("Not enough completed games in the scores sheet to build a baseline prediction.")
        st.stop()

    team_scored = team_points_scored(valid_rows, sel_team)
    team_allowed = team_points_allowed(valid_rows, sel_team)
    opp_scored = team_points_scored(valid_rows, opponent)
    opp_allowed = team_points_allowed(valid_rows, opponent)

    # Simple blend: team offense vs opponent defense & vice-versa
    # Expected points for selected team:
    team_off = np.nanmean(team_scored) if len(team_scored) else 21.0
    opp_def = np.nanmean(opp_allowed) if len(opp_allowed) else 21.0
    team_expected = float(np.nanmean([team_off, opp_def]))

    # Expected points for opponent:
    opp_off = np.nanmean(opp_scored) if len(opp_scored) else 21.0
    team_def = np.nanmean(team_allowed) if len(team_allowed) else 21.0
    opp_expected = float(np.nanmean([opp_off, team_def]))

    predicted_total = team_expected + opp_expected
    predicted_spread_team = team_expected - opp_expected  # positive => selected team projected to win by X

    st.subheader("🧮 Baseline Projection")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{sel_team} expected pts", f"{team_expected:.1f}")
    c2.metric(f"{opponent} expected pts", f"{opp_expected:.1f}")
    c3.metric("Predicted total", f"{predicted_total:.1f}")

    # Compare to your lines
    st.subheader("📏 Against Your Lines")
    total_diff = predicted_total - user_ou
    spread_diff = predicted_spread_team - (-user_spread)  # remember: user_spread is team-relative (negative=favored)
    # If user_spread = -3.5 (team favored by 3.5), the market says team -3.5.
    # Our predicted margin is predicted_spread_team. Signal = predicted - market.

    cA, cB = st.columns(2)
    with cA:
        st.write(f"**O/U {user_ou:.1f}** vs **Pred {predicted_total:.1f}** → Δ = {total_diff:+.1f}")
        st.progress(min(max((predicted_total / max(user_ou, 1.0)), 0.0), 2.0) / 2.0)
        st.write("**Lean:** " + ("Over" if total_diff > 0 else "Under") + f" (by {abs(total_diff):.1f})")

    with cB:
        # Market view: team line at user_spread ⇒ market expects margin = -user_spread for our team.
        # If predicted margin > market margin, lean to our team (cover).
        st.write(f"**Spread {sel_team} {user_spread:+.1f}** (market margin {(-user_spread):+.1f})")
        st.write(f"**Predicted margin:** {predicted_spread_team:+.1f} → Δ = {spread_diff:+.1f}")
        st.write("**Lean:** " + (f"{sel_team} to cover" if spread_diff > 0 else f"{opponent} to cover"))

    # Simple visuals
    st.subheader("📊 Visuals")
    fig_bar = px.bar(
        x=[sel_team, opponent],
        y=[team_expected, opp_expected],
        title=f"Expected Points – Week {sel_week}",
        labels={"x": "Team", "y": "Expected Points"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_total = px.bar(
        x=["Predicted Total", "Your O/U Line"],
        y=[predicted_total, user_ou],
        title="Total Points: Prediction vs Line",
        labels={"x": "", "y": "Points"}
    )
    st.plotly_chart(fig_total, use_container_width=True)

    with st.expander("ℹ️ Notes"):
        st.markdown(
            "- This is a **simple baseline** using historical scoring from your sheet only.\n"
            "- You can improve it later by blending in pace (plays/game), EPA, injuries, rest days, travel, etc."
        )
