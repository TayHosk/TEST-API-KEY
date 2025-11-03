# =========================
# NFL GAME PREDICTOR (robust loader + debug)
# =========================

NFL_SCORES_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv&gid=0"

def _normalize_header(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip().lower()
    name = name.replace(" ", "_")
    # keep alnum + underscore only
    return re.sub(r"[^0-9a-z_]", "", name)

@st.cache_data(show_spinner=False)
def load_scores_df(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # normalize headers
    df.columns = [_normalize_header(c) for c in df.columns]

    # expected cols (we’ll tolerate variants because of normalization)
    # user said these are their headers:
    # week,date,time,away_team,away_abbr,home_team,home_abbr,away_score,home_score,
    # situation,status,score_text,total_points,game_id,over_under,odds,favored_team,
    # spread,fav_covered,box_score_home,box_score_away,home_display_name,away_display_name,
    # game_winner,game_loser,over_hit,under_hit,broadcast,home_off_yards,away_off_yards
    # The normalization should match these exactly.

    # coerce dtypes that we’ll use for filters
    if "week" in df.columns:
        # Convert to int where possible
        df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")

    # Ensure team columns exist (after normalization)
    # If the sheet used slightly different names, surface that quickly:
    for col in ["home_team", "away_team"]:
        if col not in df.columns:
            # try to find close matches if needed
            candidates = [c for c in df.columns if "home" in c and "team" in c] if col == "home_team" else [c for c in df.columns if "away" in c and "team" in c]
            if candidates:
                df[col] = df[candidates[0]].astype(str)
            else:
                # create empty so downstream code doesn't crash; debug will flag it
                df[col] = np.nan

    # Build team list from union of home/away
    home_teams = df["home_team"].dropna().astype(str).str.strip().unique().tolist() if "home_team" in df.columns else []
    away_teams = df["away_team"].dropna().astype(str).str.strip().unique().tolist() if "away_team" in df.columns else []
    team_list = sorted(set(home_teams) | set(away_teams))

    # Unique, non-null weeks
    week_list = sorted([int(w) for w in df["week"].dropna().unique().tolist()]) if "week" in df.columns else []

    return df, team_list, week_list

# ---- Load & Debug panel
scores_df, team_list_gp, week_list_gp = load_scores_df(NFL_SCORES_URL)

with st.sidebar:
    st.markdown("### 🏈 Game Predictor Debug")
    st.write("Columns detected:", list(scores_df.columns))
    st.write(f"Teams detected: {len(team_list_gp)}", team_list_gp[:15], "..." if len(team_list_gp) > 15 else "")
    st.write(f"Weeks detected: {week_list_gp}")
    st.write("Head of data:")
    st.dataframe(scores_df.head(10), use_container_width=True)

# ---- UI
st.header("📈 NFL Game Predictor")

if len(team_list_gp) == 0 or len(week_list_gp) == 0:
    st.error("I couldn't find teams and/or weeks in your scoring sheet. Check the sidebar debug to verify column names and sample rows.")
    st.stop()

team_sel = st.selectbox("Select team", options=[""] + team_list_gp, index=0)
week_sel = st.selectbox("Select week", options=[""] + [str(w) for w in week_list_gp], index=0)
col1, col2 = st.columns(2)
with col1:
    user_spread = st.number_input("Your spread (fav negative, dog positive)", value=0.0, step=0.5, help="Example: -1.5 means selected team is favored by 1.5.")
with col2:
    user_ou = st.number_input("Your Over/Under line", value=44.5, step=0.5)

if not team_sel or not week_sel:
    st.info("Pick a team and week to see the matchup and prediction.")
    st.stop()

# Find the row for that team & week (either home or away)
w = int(week_sel)
week_rows = scores_df[scores_df["week"] == w]
is_home = week_rows["home_team"].astype(str).str.lower() == team_sel.lower()
is_away = week_rows["away_team"].astype(str).str.lower() == team_sel.lower()
row = week_rows[is_home | is_away]

if row.empty:
    st.warning("No game found for that team/week in the sheet.")
    st.stop()

row = row.iloc[0].copy()

team_is_home = str(row.get("home_team", "")).lower() == team_sel.lower()
opp_team = row["away_team"] if team_is_home else row["home_team"]

st.subheader(f"Matchup: {team_sel} vs {opp_team} (Week {w})")

# Pull historical/season data from sheet to construct a simple baseline model:
# We'll create a naive Poisson-like estimate based on a team's past total_points (for both home & away roles)
# and compare to opponent. If you later add richer features, plug them here.

def safe_mean(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else np.nan

# Team historical totals (season-to-date from your sheet)
team_rows = scores_df[(scores_df["home_team"].astype(str).str.lower() == team_sel.lower()) |
                      (scores_df["away_team"].astype(str).str.lower() == team_sel.lower())]

opp_rows = scores_df[(scores_df["home_team"].astype(str).str.lower() == str(opp_team).lower()) |
                     (scores_df["away_team"].astype(str).str.lower() == str(opp_team).lower())]

# Points scored by team in those games (use score_text if separate, else derive from home/away scores)
def team_points_scored(df, team):
    # Prefer explicit per-side scores if present
    if "home_score" in df.columns and "away_score" in df.columns:
        pts = []
        for _, r in df.iterrows():
            if str(r.get("home_team","")).lower() == team.lower():
                pts.append(pd.to_numeric(r.get("home_score"), errors="coerce"))
            elif str(r.get("away_team","")).lower() == team.lower():
                pts.append(pd.to_numeric(r.get("away_score"), errors="coerce"))
        s = pd.Series(pts, dtype="float")
        return s.dropna()
    # fallback: if only total points present, split evenly (very rough)
    if "total_points" in df.columns:
        return pd.to_numeric(df["total_points"], errors="coerce").dropna() / 2.0
    return pd.Series([], dtype="float")

team_pts = team_points_scored(team_rows, team_sel)
opp_pts  = team_points_scored(opp_rows, str(opp_team))

team_avg_pts = float(team_pts.mean()) if len(team_pts) else np.nan
opp_avg_pts  = float(opp_pts.mean()) if len(opp_pts) else np.nan

# Super-naive expectation: each team near their average; adjust by spread as a center shift for the selected team
# E[team] and E[opp] such that (E[team] - E[opp]) ≈ -user_spread
# Start from means; if nan, default to 21.
base_team = team_avg_pts if not np.isnan(team_avg_pts) else 21.0
base_opp  = opp_avg_pts  if not np.isnan(opp_avg_pts)  else 21.0

# shift to match spread constraint approximately
diff = base_team - base_opp
target_diff = -user_spread  # negative spread means favored (should win by abs(spread))
shift = (target_diff - diff) / 2.0
pred_team = max(0.0, base_team + shift)
pred_opp  = max(0.0, base_opp  - shift)

pred_total = pred_team + pred_opp

# Simple probabilities using Normal approx around predictions
from scipy.stats import norm as _norm
stdev_pts = 10.0  # tune later
p_cover = 1 - _norm.cdf(((-user_spread) - (pred_team - pred_opp)) / stdev_pts)
p_over  = 1 - _norm.cdf((user_ou - pred_total) / (stdev_pts * np.sqrt(2)))

st.markdown("### 🧮 Prediction")
st.write(f"**Predicted score:** {team_sel} {pred_team:.1f} — {opp_team} {pred_opp:.1f} (Total: {pred_total:.1f})")
st.write(f"**Prob cover ({team_sel} {'-' if user_spread<0 else '+'}{abs(user_spread)}):** {p_cover*100:.1f}%")
st.write(f"**Prob OVER ({user_ou}):** {p_over*100:.1f}%")
st.write(f"**Prob UNDER:** {(1-p_over)*100:.1f}%")

# Bars
bar_df = pd.DataFrame({
    "Team": [team_sel, opp_team],
    "Predicted_Points": [pred_team, pred_opp]
})
fig_gp = px.bar(bar_df, x="Team", y="Predicted_Points", title="Predicted Points")
st.plotly_chart(fig_gp, use_container_width=True)

# League weekly overs/unders visualization (optional but handy)
if "week" in scores_df.columns and "over_hit" in scores_df.columns and "under_hit" in scores_df.columns:
    tmp = scores_df.copy()
    tmp["over_hit"] = tmp["over_hit"].astype(str).str.lower().isin(["1","true","yes","y"])
    tmp["under_hit"] = tmp["under_hit"].astype(str).str.lower().isin(["1","true","yes","y"])
    agg = tmp.groupby("week")[["over_hit","under_hit"]].sum().reset_index()
    agg = agg.sort_values("week")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Overs vs Unders by Week")
        fig_ou = px.bar(agg, x="week", y=["over_hit","under_hit"], barmode="group", title="Count of Overs/Unders")
        st.plotly_chart(fig_ou, use_container_width=True)
    with c2:
        st.markdown("#### Over Rate by Week")
        # need total games per week:
        week_counts = tmp.groupby("week")["game_id"].nunique().reset_index().rename(columns={"game_id":"games"})
        over_rate = agg.merge(week_counts, on="week", how="left")
        over_rate["over_rate"] = (over_rate["over_hit"] / over_rate["games"]).fillna(0.0)
        fig_or = px.line(over_rate, x="week", y="over_rate", markers=True, title="Over Rate")
        st.plotly_chart(fig_or, use_container_width=True)
