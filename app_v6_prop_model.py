# app_v8_props_plus_games.py
# Player Props (v7.7 core you shared) + NEW Game Predictor tab

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import re

st.set_page_config(page_title="NFL Props + Game Predictor (v8)", layout="wide")

# -------------------------------
# 1) Google Sheets
# -------------------------------
SHEETS = {
    # --- your existing sheets (unchanged) ---
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
    # --- NEW: scoring/games sheet for Game Predictor tab ---
    "games_scores": "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv",
}

# -------------------------------
# 2) Data Loaders
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
    # common normalizations
    for col in ("team", "teams", "home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_all():
    return {name: load_and_clean(url) for name, url in SHEETS.items()}

data = load_all()

# Unpack the data used by Props
p_rec  = data["player_receiving"]
p_rush = data["player_rushing"]
p_pass = data["player_passing"]
d_rb   = data["def_rb"]
d_qb   = data["def_qb"]
d_wr   = data["def_wr"]
d_te   = data["def_te"]

# Unpack the scoring/games sheet for Game Predictor
games  = data["games_scores"]

# -------------------------------
# 3) Shared Helpers
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
# 4) Layout: Tabs
# -------------------------------
tab_props, tab_games = st.tabs(["📈 Player Props", "🧮 Game Predictor"])

# =========================================================
# ===============  TAB 1: PLAYER PROPS  ===================
# =========================================================
with tab_props:
    # Sidebar Debug for props only
    with st.sidebar:
        st.header("🔎 Debug (Props)")
        st.write("Receiving cols:", list(p_rec.columns))
        st.write("Rushing cols:", list(p_rush.columns))
        st.write("Passing cols:", list(p_pass.columns))
        st.write("Def RB cols:", list(d_rb.columns))
        st.write("Def WR cols:", list(d_wr.columns))
        st.write("Def TE cols:", list(d_te.columns))
        st.write("Note: v7.7 adds dropdowns for player + team selection")

    st.title("🏈 NFL Player Prop Model (v7.7)")

    # combine player list from all sheets
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

    if player_name and opponent_team and selected_props:
        st.header("📊 Results")

        for prop in selected_props:
            # --- ANYTIME TD (Rushing + Receiving Combined) ---
            if prop == "anytime_td":
                st.subheader("🔥 Anytime TD (Rushing + Receiving + Defense Adjusted)")

                rec_row = find_player_in(p_rec, player_name)
                rush_row = find_player_in(p_rush, player_name)

                total_tds = 0.0
                total_games = 0.0

                for dfp in [rec_row, rush_row]:
                    if dfp is not None and not dfp.empty:
                        td_cols = [c for c in dfp.columns if "td" in c and "allowed" not in c]
                        games_col = "games_played" if "games_played" in dfp.columns else None
                        if td_cols and games_col:
                            tds = sum(float(dfp.iloc[0][col]) for col in td_cols if pd.notna(dfp.iloc[0][col]))
                            total_tds += tds
                            total_games = max(total_games, float(dfp.iloc[0][games_col]))

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
                    d.loc[d["team"].astype(str).str.lower() == opponent_team.lower(), "tds_pg"].mean()
                    for d in def_dfs
                ])
                if np.isnan(opp_td_pg):
                    opp_td_pg = league_td_pg

                adj_factor = opp_td_pg / league_td_pg if league_td_pg and league_td_pg > 0 else 1.0
                adj_td_rate = player_td_pg * adj_factor
                prob_anytime = float(np.clip(adj_td_rate, 0.001, 0.999))

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

            # --- OTHER PROPS (your working logic) ---
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
                player_df = (find_player_in(p_rec, player_name)
                             or find_player_in(p_rush, player_name)
                             or find_player_in(p_pass, player_name))
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

            adj_factor = (opp_allowed_pg / league_allowed_pg) if (league_allowed_pg and league_allowed_pg > 0) else 1.0
            predicted_pg = player_pg * adj_factor

            line_val = float(lines.get(prop, 0.0))
            stdev = max(3.0, predicted_pg * 0.35)
            z = (line_val - predicted_pg) / stdev
            prob_over = float(np.clip(1 - norm.cdf(z), 0.001, 0.999))
            prob_under = float(np.clip(norm.cdf(z), 0.001, 0.999))

            st.subheader(prop.replace("_", " ").title())
            st.write(f"**Player (season):** {season_val:.2f} over {games_played:.0f} games → **{player_pg:.2f} per game**")
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

    else:
        st.info("Select a player, opponent team, and at least one prop to run the model.")

# =========================================================
# ===============  TAB 2: GAME PREDICTOR  =================
# =========================================================
with tab_games:
    st.title("🧮 NFL Game Predictor (Season View)")

    # Expecting normalized columns like:
    # week, date, time, away_team, home_team, away_score, home_score, ou, spread, favored_team, fav_covered, over_flag, under_flag, home_off_yds, away_off_yds, ...
    # We'll try to infer boolean-ish fields
    def to_bool_series(s: pd.Series):
        # Accepts True/False, 1/0, "true"/"false", "yes"/"no", "over"/"under"
        return s.astype(str).str.strip().str.lower().map({
            "true": True, "t": True, "1": True, "yes": True, "y": True, "over": True,
            "false": False, "f": False, "0": False, "no": False, "n": False, "under": False
        })

    # Coerce types
    for col in ["week", "away_score", "home_score", "ou", "spread", "home_off_yds", "away_off_yds"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")

    # Flags
    if "over_flag" in games.columns:
        games["over_flag"] = to_bool_series(games["over_flag"])
    else:
        # fallback: compute from totals if we have ou and scores
        if set(["ou", "away_score", "home_score"]).issubset(games.columns):
            games["total_points"] = games["away_score"] + games["home_score"]
            games["over_flag"] = games["total_points"] > games["ou"]
        else:
            games["over_flag"] = False

    if "under_flag" in games.columns:
        games["under_flag"] = to_bool_series(games["under_flag"])
    else:
        games["under_flag"] = ~games["over_flag"]

    if "fav_covered" in games.columns:
        games["fav_covered"] = to_bool_series(games["fav_covered"])
    else:
        games["fav_covered"] = None

    # League-wide Over/Under chart by week
    st.subheader("📊 League: Over vs Under by Week")
    if "week" in games.columns:
        ou_week = games.groupby("week").agg(
            overs=("over_flag", "sum"),
            unders=("under_flag", "sum"),
            games=("over_flag", "count")
        ).reset_index()
        ou_week = ou_week.sort_values("week")
        fig_ou = px.bar(
            ou_week.melt(id_vars="week", value_vars=["overs", "unders"], var_name="Outcome", value_name="Count"),
            x="week", y="Count", color="Outcome",
            barmode="group",
            title="Over vs Under Counts by Week"
        )
        st.plotly_chart(fig_ou, use_container_width=True)
    else:
        st.warning("No 'week' column found in scoring sheet.")

    st.markdown("---")

    # Team-specific analysis
    st.subheader("🏟️ Team Weekly Totals vs O/U")
    # Team list from home/away
    team_opts = sorted(set(list(games.get("home_team", pd.Series(dtype=str)).dropna().unique())
                           + list(games.get("away_team", pd.Series(dtype=str)).dropna().unique())))
    team_choice = st.selectbox("Select Team", options=[""] + team_opts, index=0)

    if team_choice:
        # Filter team games
        mask_home = games.get("home_team", "").astype(str).str.lower() == team_choice.lower()
        mask_away = games.get("away_team", "").astype(str).str.lower() == team_choice.lower()
        tg = games[mask_home | mask_away].copy()

        if tg.empty:
            st.warning("No games found for that team in the sheet.")
        else:
            # Compute totals and over/under vs O/U line
            tg["total_points"] = tg.get("home_score", 0) + tg.get("away_score", 0)
            tg = tg.sort_values("week").reset_index(drop=True)

            # team points scored/allowed
            def team_points(row):
                if str(row.get("home_team","")).lower() == team_choice.lower():
                    return row.get("home_score", np.nan), row.get("away_score", np.nan)
                else:
                    return row.get("away_score", np.nan), row.get("home_score", np.nan)

            pts = tg.apply(team_points, axis=1, result_type="expand")
            tg["team_scored"] = pts[0]
            tg["team_allowed"] = pts[1]

            # spread cover for the selected team
            # rule: team_covered = (favored_team==team & fav_covered True) OR (favored_team!=team & fav_covered False)
            if {"favored_team", "fav_covered"}.issubset(tg.columns):
                tg["team_covered"] = np.where(
                    tg["favored_team"].astype(str).str.lower() == team_choice.lower(),
                    tg["fav_covered"] == True,
                    tg["fav_covered"] == False
                )
            else:
                tg["team_covered"] = np.nan

            # Summary
            avg_scored = float(np.nanmean(tg["team_scored"]))
            avg_allowed = float(np.nanmean(tg["team_allowed"]))
            over_pct = float(100 * np.nanmean(tg["over_flag"])) if "over_flag" in tg.columns else np.nan
            cover_pct = float(100 * np.nanmean(tg["team_covered"])) if "team_covered" in tg.columns else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Points Scored", f"{avg_scored:.1f}")
            c2.metric("Avg Points Allowed", f"{avg_allowed:.1f}")
            c3.metric("Over %", f"{over_pct:.1f}%" if not np.isnan(over_pct) else "n/a")
            c4.metric("Cover %", f"{cover_pct:.1f}%" if not np.isnan(cover_pct) else "n/a")

            # Line chart: team weekly total vs O/U
            line_df = tg.copy()
            # safe OU
            if "ou" not in line_df.columns:
                line_df["ou"] = np.nan
            # label for color (Over/Under/Push)
            def ou_label(row):
                if pd.isna(row["ou"]) or pd.isna(row["total_points"]):
                    return "n/a"
                if row["total_points"] > row["ou"]:
                    return "Over"
                elif row["total_points"] < row["ou"]:
                    return "Under"
                return "Push"

            line_df["ou_result"] = line_df.apply(ou_label, axis=1)

            # melt for plotting
            plot_long = line_df[["week", "total_points", "ou"]].melt(id_vars="week", var_name="Series", value_name="Value")
            fig_team = px.line(
                plot_long.dropna(subset=["Value"]),
                x="week", y="Value", color="Series", markers=True,
                title=f"{team_choice} – Weekly Total Points vs O/U"
            )
            # add color dots for Over/Under classification
            dots = line_df.dropna(subset=["total_points"])
            fig_dots = px.scatter(
                dots, x="week", y="total_points", color="ou_result",
                title=None
            )
            for t in fig_dots.data:
                fig_team.add_trace(t)

            st.plotly_chart(fig_team, use_container_width=True)

    else:
        st.info("Pick a team to see week-by-week totals vs O/U, plus summary stats.")
