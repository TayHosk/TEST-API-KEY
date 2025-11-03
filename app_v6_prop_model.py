# =========================================================
# 📈 NFL GAME PREDICTOR TAB
# =========================================================

import plotly.express as px

st.sidebar.markdown("---")
page = st.sidebar.radio("Select Page", ["🏈 Player Prop Model", "📈 NFL Game Predictor"], index=0)

if page == "📈 NFL Game Predictor":
    st.title("📈 NFL Game Predictor")

    # ---- Load the NFL scoring sheet ----
    NFL_SCORES_URL = "https://docs.google.com/spreadsheets/d/1KrTQbR5uqlBn2v2Onpjo6qHFnLlrqIQBzE52KAhMYcY/export?format=csv&gid=0"

    @st.cache_data(show_spinner=False)
    def load_scores_df(url):
        df = pd.read_csv(url)
        df.columns = [normalize_header(c) for c in df.columns]
        if "week" in df.columns:
            df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
        return df

    try:
        scores_df = load_scores_df(NFL_SCORES_URL)
    except Exception as e:
        st.error(f"Error loading NFL data: {e}")
        st.stop()

    if scores_df.empty:
        st.warning("⚠️ No data loaded from the Google Sheet.")
        st.stop()

    # ---- Dropdown setup ----
    teams = sorted(
        list(scores_df["home_team"].dropna().unique()) +
        list(scores_df["away_team"].dropna().unique())
    )
    weeks = sorted([int(w) for w in scores_df["week"].dropna().unique()])

    team_sel = st.selectbox("Select Team", [""] + teams)
    week_sel = st.selectbox("Select Week", [""] + [str(w) for w in weeks])
    spread = st.number_input("Spread (negative = favorite)", value=-2.5)
    over_under = st.number_input("Over/Under", value=44.5)

    if not team_sel or not week_sel:
        st.stop()

    week_sel = int(week_sel)

    # ---- Find that week's game ----
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

    # ---- Calculate average scoring ----
    def avg_points(df, team):
        pts = []
        for _, r in df.iterrows():
            if r["home_team"].lower() == team.lower():
                pts.append(r["home_score"])
            elif r["away_team"].lower() == team.lower():
                pts.append(r["away_score"])
        return np.nanmean(pd.to_numeric(pts, errors="coerce"))

    team_games = scores_df[
        (scores_df["home_team"].str.lower() == team_sel.lower()) |
        (scores_df["away_team"].str.lower() == team_sel.lower())
    ]
    opp_games = scores_df[
        (scores_df["home_team"].str.lower() == opponent.lower()) |
        (scores_df["away_team"].str.lower() == opponent.lower())
    ]

    team_avg = avg_points(team_games, team_sel) or 21
    opp_avg = avg_points(opp_games, opponent) or 21

    # ---- Prediction logic ----
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
