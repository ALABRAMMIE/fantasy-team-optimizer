import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

st.title("Fantasy Team Optimizer")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df)

    st.sidebar.header("Constraints")
    budget = st.sidebar.number_input("Max Budget", value=140.0)
    team_size = st.sidebar.number_input("Team Size", value=13, step=1)

    include_players = st.sidebar.multiselect("Players to Include", df["Name"])
    exclude_players = st.sidebar.multiselect("Players to Exclude", df["Name"])

    if st.button("Optimize Team"):
        players = df.to_dict("records")

        prob = LpProblem("FantasyTeam", LpMaximize)

        # Create binary decision variables
        x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

        # Objective: Maximize total FTPS
        prob += lpSum(x[p["Name"]] * p["FTPS"] for p in players)

        # Budget constraint
        prob += lpSum(x[p["Name"]] * p["Value"] for p in players) <= budget

        # Team size constraint
        prob += lpSum(x[p["Name"]] for p in players) == team_size

        # Include constraints
        for name in include_players:
            prob += x[name] == 1

        # Exclude constraints
        for name in exclude_players:
            prob += x[name] == 0

        # Solve
        prob.solve()

        selected = [p for p in players if x[p["Name"]].value() == 1]

        st.subheader("Optimized Team")
        result_df = pd.DataFrame(selected)
        st.dataframe(result_df)

        total_value = sum(p["Value"] for p in selected)
        total_ftps = sum(p["FTPS"] for p in selected)

        st.write(f"**Total Value**: {total_value}")
        st.write(f"**Total FTPS**: {total_ftps}")

        st.download_button("Download Team as CSV", result_df.to_csv(index=False), file_name="optimized_team.csv")
