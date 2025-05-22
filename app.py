
import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random

st.title("Fantasy Team Optimizer")

sport_options = [
    "-- Choose a sport --",
    "Cycling", "Speed Skating", "Formula 1", "Stock Exchange", "Tennis", "MotoGP", "Football",
    "Darts", "Cyclocross", "Golf", "Snooker", "Olympics", "Basketball", "Dakar Rally", "Skiing",
    "Rugby", "Biathlon", "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment"
]

sport = st.sidebar.selectbox("Select a sport", sport_options)

budget = st.sidebar.number_input("Max Budget", value=100.0)
team_size = st.sidebar.number_input("Team Size", value=11, step=1)

solver_mode = st.sidebar.radio("Solver Objective", [
    "Maximize Budget Usage",
    "Closest FTP Match"
])

uploaded_file = st.file_uploader("Upload your Excel file (players)", type=["xlsx"])
template_file = st.sidebar.file_uploader("Upload Target Profile Template (multi-sheet)", type=["xlsx"], key="multi-template")

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if not {"Name", "Value"}.issubset(df.columns):
            st.error("Uploaded file must include at least 'Name' and 'Value' columns.")
        else:
            st.subheader("📋 Edit Player Data")
            edited_df = st.data_editor(df[["Name", "Value"]], use_container_width=True)
            players = edited_df.to_dict("records")

            if solver_mode == "Closest FTP Match":
                if not template_file:
                    st.warning("Please upload a target profile template for Closest FTP Match.")
                else:
                    try:
                        profile_sheet = pd.read_excel(template_file, sheet_name=sport, header=None)
                        raw_values = profile_sheet.iloc[:, 0].dropna().astype(float).tolist()
                        if len(raw_values) < team_size:
                            st.error(f"Target profile for {sport} has fewer than {team_size} values.")
                        else:
                            target_values = raw_values[:team_size]

                            selected_team = []
                            used_names = set()
                            for target in target_values:
                                candidates = sorted(
                                    [p for p in players if p["Name"] not in used_names],
                                    key=lambda p: abs(p["Value"] - target)
                                )
                                for p in candidates:
                                    if p["Name"] not in used_names:
                                        selected_team.append(p)
                                        used_names.add(p["Name"])
                                        break

                            if len(selected_team) == team_size:
                                result_df = pd.DataFrame(selected_team)
                                st.subheader("🎯 Closest FTP Match Team")
                                st.dataframe(result_df)
                                st.write(f"**Total Value**: {sum(p['Value'] for p in selected_team)}")
                                st.download_button("📥 Download Team", result_df.to_csv(index=False), file_name="closest_match_team.csv")
                            else:
                                st.error("Could not form a complete team.")
                    except Exception as e:
                        st.error(f"Failed to read profile for sport '{sport}': {e}")
            else:
                # Maximize Budget Usage
                prob = LpProblem("FantasyTeam", LpMaximize)
                x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
                prob += lpSum(x[p["Name"]] * p["Value"] for p in players)
                prob += lpSum(x[p["Name"]] for p in players) == team_size
                prob += lpSum(x[p["Name"]] * p["Value"] for p in players) <= budget
                prob.solve()

                selected = [p for p in players if x[p["Name"]].value() == 1]
                if len(selected) == team_size:
                    result_df = pd.DataFrame(selected)
                    st.subheader("💰 Maximize Budget Usage Team")
                    st.dataframe(result_df)
                    st.write(f"**Total Value**: {sum(p['Value'] for p in selected)}")
                    st.download_button("📥 Download Team", result_df.to_csv(index=False), file_name="max_budget_team.csv")
                else:
                    st.error("Couldn't form a valid team under constraints.")
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
