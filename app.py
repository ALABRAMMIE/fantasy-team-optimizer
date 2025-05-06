import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

st.title("Fantasy Team Optimizer")

# English-only sport list
sport_options = [
    "-- Choose a sport --",
    "Cycling", "Speed Skating", "Formula 1", "Stock Exchange", "Tennis", "MotoGP", "Football",
    "Darts", "Cyclocross", "Golf", "Snooker", "Olympics", "Basketball", "Dakar Rally", "Skiing",
    "Rugby", "Biathlon", "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment"
]

# Sport selector
sport = st.sidebar.selectbox("Select a sport", sport_options)

if sport == "-- Choose a sport --":
    st.info("Please select a sport to begin.")
elif sport == "Cycling":
    st.sidebar.header("🚴 Cycling Constraints")
    budget = st.sidebar.number_input("Max Budget", value=140.0)
    team_size = st.sidebar.number_input("Team Size", value=13, step=1)

    st.sidebar.markdown("---")
    uploaded_file = st.file_uploader("Upload your Excel file for Cycling", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.subheader(f"📄 Uploaded data for: {sport}")
        st.dataframe(df)

        include_players = st.sidebar.multiselect("Players to INCLUDE", df["Name"])
        exclude_players = st.sidebar.multiselect("Players to EXCLUDE", df["Name"])

        if st.button("Optimize Cycling Team"):
            players = df.to_dict("records")

            prob = LpProblem("FantasyTeam", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            prob += lpSum(x[p["Name"]] * p["FTPS"] for p in players)
            prob += lpSum(x[p["Name"]] * p["Value"] for p in players) <= budget
            prob += lpSum(x[p["Name"]] for p in players) == team_size

            for name
