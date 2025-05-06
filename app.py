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

# Select a sport
sport = st.sidebar.selectbox("Select a sport", sport_options)

if sport == "-- Choose a sport --":
    st.info("Please select a sport to begin.")
elif sport == "Cycling":
    uploaded_file = st.file_uploader("Upload your Excel file for Cycling", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.subheader(f"📄 Uploaded data for: {sport}")
        st.dataframe(df)

        st.sidebar.header("🚴 Cycling Constraints")
        budget = st.sidebar.number_input("Max Budget", value=140.0)
        team_size = st.sidebar.number_input("Team Size", value=13, step=1)

        include_players = st.sidebar.multiselect("Players to INCLUDE", df["Name"])
        exclude_players = st.sidebar.multiselect("Players to EXCLUDE", df["Name"])

        if st.button("Optimize Cycling Team"):
            players = df.to_dict("records")

            prob = LpProblem("FantasyTeam", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            prob += lpSum(x[p["Name"]] * p["FTPS"] for p in players)
            prob += lpSum(x[p["Name"]] * p["Value"] for p in players) <= budget
            prob += lpSum(x[p["Name"]] for p in players) == team_size

            for name in include_players:
                prob += x[name] == 1
            for name in exclude_players:
                prob += x[name] == 0

            prob.solve()
            selected = [p for p in players if x[p["Name"]].value() == 1]

            st.subheader("✅ Optimized Cycling Team")
            result_df = pd.DataFrame(selected)
            st.dataframe(result_df)

            st.write(f"**Total Value**: {sum(p['Value'] for p in selected)}")
            st.write(f"**Total FTPS**: {sum(p['FTPS'] for p in selected)}")

            st.download_button("📥 Download Team as CSV", result_df.to_csv(index=False), file_name="optimized_team.csv")
else:
    st.warning(f"The constraint system for **{sport}** is not configured yet. Coming soon!")




