import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

st.title("Fantasy Team Optimizer")

# English-only sport options
sport_options = [
    "-- Choose a sport --",
    "Cycling", "Speed Skating", "Formula 1", "Stock Exchange", "Tennis", "MotoGP", "Football",
    "Darts", "Cyclocross", "Golf", "Snooker", "Olympics", "Basketball", "Dakar Rally", "Skiing",
    "Rugby", "Biathlon", "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment"
]

# Sport selector
sport = st.sidebar.selectbox("Select a sport", sport_options)

# Show optimizer UI only when a valid sport is selected
if sport != "-- Choose a sport --":
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.subheader(f"📄 Uploaded data for: {sport}")
        st.dataframe(df)

        st.sidebar.header("⚙️ Settings")
        budget = st.sidebar.number_input("Max Budget", value=140.0)
        team_size = st.sidebar.number_input("Team Size", value=13, step=1)

        include_players = st.sidebar.multiselect("Players to INCLUDE", df["Name"])
        exclude_players = st.sidebar.multiselect("Players to EXCLUDE", df["Name"])

        if st.button("Optimize Team"):
            players = df



