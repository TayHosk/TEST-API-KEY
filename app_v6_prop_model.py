# ================================
# NFL GAME PREDICTOR (Calibrated)
# ================================
# Requirements:
# - scores_csv_url in SHEETS like you set earlier (the one with these headers)
#   week,date,time,away_team,away_abbr,home_team,home_abbr,away_score,home_score,
#   situation,status,score_text,total_points,game_id,over_under,odds,favored_team,
#   spread,fav_covered,box_score_home,box_score_away,home_display_name,away_display_name,
#   game_winner,game_loser,over_hit,under_hit,broadcast,home_off_yards,away_off_yards
#
# This tab reads the CSV directly (no Google auth), computes a baseline,
# then calibrates to Vegas lines from prior weeks to match market levels.

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm

st.markdown("## 🧮 NFL Game Predictor (Calibrated to Market)")
scores_csv_url = st.text_input(
    "Scores CSV URL (export=csv)",
    "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv"
)

@st.cache_data(show_spinner=False)
def load_scores(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    # normalize headers
    def norm(s): 
        s = str(s).strip().lower().replace(" ", "_")
        return "".join(ch for ch in s if ch.isalnum() or ch == "_")
    df.columns = [norm(c) for c in df.columns]
    # best-effort types
    for c in ["week","away_score","home_score","total_points","over_under","spread","home_off_yards","away_off_yards"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # teams to str
    for c in ["home_team","away_team","home_abbr","away_abbr"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

scores_df = load_scores(scores_csv_url)

if scores_df.empty:
    st.warning("Couldn't load the scores sheet. Double-check the URL (must be a CSV export).")
    st.stop()

# --- UI: pick week and team
weeks = sorted([int(w) for w in scores_df["week"].dropna().unique() if pd.notna(w)])
sel_week = st.selectbox("Select Week", options=weeks, index=(len(weeks)-1 if weeks else 0))

# Collect team universe from either abbr or team fields
team_pool = set()
for col in ["home_abbr","away_abbr","home_team","away_team"]:
    if col in scores_df.columns:
        team_pool.update(scores_df[col].dropna().astype(str).unique().tolist())
teams = sorted([t for t in team_pool if t and t.lower() != "nan"])

sel_team = st.selectbox("Select Your Team", options=[""] + teams, index=0, help="Pick the team you want to project")
if not sel_team:
    st.info("Pick a team to continue.")
    st.stop()

# Enter live market lines you want to compare against
col1, col2 = st.columns(2)
with col1:
    user_spread = st.number_input("Enter spread (negative = your team favored)", value= -2.5, step=0.5, format="%.2f")
with col2:
    user_total  = st.number_input("Enter total (O/U)", value= 44.5, step=0.5, format="%.1f")

# --- Helper: build per-team seasonal averages up to (but not including) sel_week
def season_team_avgs(df: pd.DataFrame, week_cutoff: int):
    hist = df[(df["week"].notna()) & (df["week"] < week_cutoff)].copy()
    # rows with valid scores only
    hist = hist[pd.notna(hist["home_score"]) & pd.notna(hist["away_score"])]

    # long form
    home = hist[["home_team","home_abbr","home_score","away_score"]].copy()
    home["team"] = home["home_abbr"].where(home["home_abbr"].notna(), home["home_team"])
    home["pts_for"]  = home["home_score"]
    home["pts_again"] = home["away_score"]

    away = hist[["away_team","away_abbr","home_score","away_score"]].copy()
    away["team"] = away["away_abbr"].where(away["away_abbr"].notna(), away["away_team"])
    away["pts_for"]  = away["away_score"]
    away["pts_again"] = away["home_score"]

    long = pd.concat([home[["team","pts_for","pts_again"]],
                      away[["team","pts_for","pts_again"]]], ignore_index=True)

    grp = long.groupby("team", dropna=True).agg(
        games=("pts_for","count"),
        avg_scored=("pts_for","mean"),
        avg_allowed=("pts_again","mean"),
    ).reset_index()

    # fallback for teams with no history yet
    league_avg_scored  = long["pts_for"].mean() if not long.empty else 22.0
    league_avg_allowed = long["pts_again"].mean() if not long.empty else 22.0

    grp["avg_scored"]  = grp["avg_scored"].fillna(league_avg_scored)
    grp["avg_allowed"] = grp["avg_allowed"].fillna(league_avg_allowed)

    return grp, league_avg_scored, league_avg_allowed

team_avgs, league_pts_for, league_pts_again = season_team_avgs(scores_df, sel_week)

# --- Find opponent this week (from schedule)
wk = scores_df[scores_df["week"] == sel_week].copy()

def as_team_str(x):
    return str(x).strip()

wk["home"] = wk["home_abbr"].fillna(wk["home_team"]).apply(as_team_str)
wk["away"] = wk["away_abbr"].fillna(wk["away_team"]).apply(as_team_str)

mask_is_home = wk["home"].str.lower() == sel_team.lower()
mask_is_away = wk["away"].str.lower() == sel_team.lower()

if not (mask_is_home.any() or mask_is_away.any()):
    st.warning(f"I couldn't find **{sel_team}** in Week {sel_week} of your sheet.")
    st.stop()

row = wk[mask_is_home | mask_is_away].iloc[0]
is_home = bool(mask_is_home.any())
opp    = row["away"] if is_home else row["home"]

st.markdown(f"**Matchup (Week {sel_week}):** {sel_team} {'vs' if is_home else '@'} {opp}")

# --- Baseline (un-calibrated) projection
def get_avg(team_name: str, col: str, default: float):
    s = team_avgs.loc[team_avgs["team"].astype(str).str.lower()==team_name.lower(), col]
    return float(s.iloc[0]) if not s.empty and pd.notna(s.iloc[0]) else default

team_avg_scored  = get_avg(sel_team, "avg_scored",  league_pts_for)
team_avg_allowed = get_avg(sel_team, "avg_allowed", league_pts_again)
opp_avg_scored   = get_avg(opp,      "avg_scored",  league_pts_for)
opp_avg_allowed  = get_avg(opp,      "avg_allowed", league_pts_again)

# Baseline (your earlier logic): simple average of offense vs defense
raw_team_pts = (team_avg_scored + opp_avg_allowed) / 2.0
raw_opp_pts  = (opp_avg_scored  + team_avg_allowed) / 2.0
raw_total    = raw_team_pts + raw_opp_pts
raw_spread   = raw_team_pts - raw_opp_pts  # (your team) - (opp)

# --- Fit calibration using past games (<= sel_week-1)
hist = scores_df[(scores_df["week"].notna()) & (scores_df["week"] < sel_week)].copy()
hist = hist[pd.notna(hist["home_score"]) & pd.notna(hist["away_score"])]

def compute_raw_pair(hrow):
    # compute same raw model for any game row
    hteam = as_team_str(hrow["home_abbr"]) if pd.notna(hrow.get("home_abbr", None)) else as_team_str(hrow["home_team"])
    ateam = as_team_str(hrow["away_abbr"]) if pd.notna(hrow.get("away_abbr", None)) else as_team_str(hrow["away_team"])
    h_scored  = get_avg(hteam, "avg_scored",  league_pts_for)
    h_allowed = get_avg(hteam, "avg_allowed", league_pts_again)
    a_scored  = get_avg(ateam, "avg_scored",  league_pts_for)
    a_allowed = get_avg(ateam, "avg_allowed", league_pts_again)
    h_raw = (h_scored + a_allowed) / 2.0
    a_raw = (a_scored + h_allowed) / 2.0
    return h_raw + a_raw, h_raw - a_raw

hist["actual_total"] = hist["home_score"] + hist["away_score"]
raw_pairs = hist.apply(lambda r: compute_raw_pair(r), axis=1, result_type="expand")
hist["raw_total"]  = raw_pairs[0]
hist["raw_spread"] = raw_pairs[1]

# We’ll regress actual_total ~ raw_total and actual_spread ~ raw_spread
def fit_calibration(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 15:  # not enough games -> fallback
        return 0.0, 1.10, 7.0  # intercept, slope, sigma fallback
    # linear fit y = a + b*x
    a, b = np.polyfit(x[mask], y[mask], 1)
    # std of residuals for uncertainty
    resid = y[mask] - (a + b*x[mask])
    sigma = float(np.sqrt(np.mean(resid**2)))
    return float(a), float(b), sigma

a_tot, b_tot, sigma_tot = fit_calibration(hist["raw_total"], hist["actual_total"])

# For spreads we need actual spread (home - away)
hist["actual_spread"] = hist["home_score"] - hist["away_score"]
a_spr, b_spr, sigma_spr = fit_calibration(hist["raw_spread"], hist["actual_spread"])

# --- Apply calibration
cal_total  = a_tot + b_tot * raw_total
cal_spread = a_spr + b_spr * raw_spread

# Split calibrated total using raw ratio (to keep offense/def mix)
raw_team_share = raw_team_pts / max(raw_total, 1e-6)
raw_opp_share  = 1.0 - raw_team_share
cal_team_pts   = cal_total * raw_team_share
cal_opp_pts    = cal_total * raw_opp_share

# Round to plausible football scores for display
disp_team = float(np.round(cal_team_pts, 1))
disp_opp  = float(np.round(cal_opp_pts, 1))
disp_total = float(np.round(cal_total, 1))
disp_spread = float(np.round(cal_spread, 1))

st.markdown("### 🔧 Calibrated Projection (closer to market)")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{sel_team} points", disp_team)
c2.metric(f"{opp} points", disp_opp)
c3.metric("Projected Total", disp_total)
c4.metric(f"Projected Spread ( {sel_team} - {opp} )", disp_spread)

# --- Compare to your lines (edge + rough probabilities)
def prob_over_under(line_total, pred_total, sigma):
    if sigma <= 0: sigma = 6.5  # safety
    z = (line_total - pred_total) / sigma
    p_over = 1 - norm.cdf(z)
    return float(np.clip(p_over, 0.01, 0.99)), float(1 - np.clip(p_over, 0.01, 0.99))

def prob_cover(line_spread, pred_spread, sigma):
    if sigma <= 0: sigma = 5.5
    # P( pred_spread > line_spread )
    z = (line_spread - pred_spread) / sigma
    p_cover = 1 - norm.cdf(z)
    return float(np.clip(p_cover, 0.01, 0.99))

p_over, p_under = prob_over_under(user_total, cal_total, sigma_tot)
p_cover = prob_cover(user_spread, cal_spread, sigma_spr)

edge_total = float(np.round(cal_total - user_total, 2))
edge_spread = float(np.round(cal_spread - user_spread, 2))

st.markdown("### 📈 Edges vs Your Line")
e1, e2 = st.columns(2)
with e1:
    st.write(f"**Total edge (model − line):** {edge_total:+.2f}")
    st.write(f"**Prob OVER {user_total:.1f}:** {p_over*100:.1f}%  |  **Prob UNDER:** {p_under*100:.1f}%")
with e2:
    st.write(f"**Spread edge (model − line):** {edge_spread:+.2f}")
    lab = f"{sel_team} {'to cover' if user_spread<0 else 'to beat the +spread'}"
    st.write(f"**Prob {lab}:** {p_cover*100:.1f}%")

# --- Visualization
st.markdown("### 🔭 Visualization")
fig = px.bar(
    x=[f"{sel_team}", f"{opp}"],
    y=[disp_team, disp_opp],
    title=f"Projected Score – Week {sel_week}: {sel_team} vs {opp}",
    labels={"x":"Team","y":"Points"}
)
st.plotly_chart(fig, use_container_width=True)

# --- Debug (optional)
with st.expander("Model details / calibration debug"):
    st.write(f"Raw model: team={raw_team_pts:.2f}, opp={raw_opp_pts:.2f}, total={raw_total:.2f}, spread={raw_spread:.2f}")
    st.write(f"Calibration: total = {a_tot:+.2f} + {b_tot:.3f}·raw (σ≈{sigma_tot:.2f})")
    st.write(f"Calibration: spread = {a_spr:+.2f} + {b_spr:.3f}·raw (σ≈{sigma_spr:.2f})")
