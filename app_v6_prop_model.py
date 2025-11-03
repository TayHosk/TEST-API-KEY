import streamlit as st

st.set_page_config(page_title="NFL App", layout="wide")

# MUST be the first Streamlit UI call
page = st.sidebar.radio("Select Page", ["🏈 Player Prop Model", "📈 NFL Game Predictor"])

if page == "🏈 Player Prop Model":
    st.title("🏈 Player Prop Model")
    st.write("This is your working props tab. Replace this with your existing code.")

if page == "📈 NFL Game Predictor":
    st.title("📈 NFL Game Predictor")
    st.write("This is your new game predictor tab.")
