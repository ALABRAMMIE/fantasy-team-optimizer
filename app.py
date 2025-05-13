import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum
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

if sport == "-- Choose a sport --":
    st.info("Please select a sport to begin.")
elif sport == "Cycling":
    st.sidebar.header("🚴 Cycling Constraints")
    budget = st.sidebar.number_input("Max Budget", value=140.0)
    team_size = st.sidebar.number_input("Team Size", value=13, step=1)

    solver_mode = st.sidebar.radio("Solver Objective", [
        "Maximize FTPS",
        "Maximize Budget Usage",
        "Match Winning FTPS Profile",
        "Closest FTP Match"
    ])

    uploaded_file = st.file_uploader("Upload your Cycling Excel file", type=["xlsx"])
    template_file = st.sidebar.file_uploader("Upload Historic Winners Template", type=["xlsx"], key="template")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        required_cols = {"Name", "Value"}
        if not required_cols.issubset(df.columns):
            st.error(f"Your file must include at least: {', '.join(required_cols)}")
        else:
            st.subheader("📋 Edit Player Data (Rank FTPS is editable)")
            editable_cols = ["Name", "Value"]
            if "Position" in df.columns:
                editable_cols.append("Position")
            if "Rank FTPS" in df.columns:
                editable_cols.append("Rank FTPS")

            edited_df = st.data_editor(df[editable_cols], use_container_width=True)

            if solver_mode != "Maximize Budget Usage" and "Rank FTPS" in edited_df.columns:
                default_rank_points = {rank: max(0, 150 - (rank - 1) * 5) for rank in range(1, 31)}
                edited_df["FTPS"] = edited_df["Rank FTPS"].apply(
                    lambda r: default_rank_points.get(int(r), 0) if pd.notnull(r) else 0
                )
            elif solver_mode == "Maximize FTPS":
                st.warning("⚠️ FTPS optimization selected but 'Rank FTPS' column is missing.")
                edited_df["FTPS"] = 0

            include_players = st.sidebar.multiselect("Players to INCLUDE", edited_df["Name"])
            exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited_df["Name"])
            optimize_clicked = st.sidebar.button("🚀 Optimize Cycling Team")

            if optimize_clicked:
                players = edited_df.to_dict("records")
                target_values = None

                if solver_mode in ["Match Winning FTPS Profile", "Closest FTP Match"] and template_file:
                    try:
                        profile_template = pd.read_excel(template_file, header=None)
                        raw_values = profile_template.iloc[1:14, 2].astype(float).values  # C2:C14

                        original_total = sum(raw_values)
                        percentages = [v / original_total for v in raw_values]
                        target_values = [p * budget for p in percentages]

                        st.write("📊 Target Values Scaled to Budget:", target_values)
                    except Exception as e:
                        st.error(f"Failed to process template: {e}")

                if solver_mode == "Closest FTP Match" and target_values:
                    available_players = [p for p in players if p["Name"] not in exclude_players]
                    selected_team = []
                    used_names = set()
                    running_value_total = 0.0

                    for target in target_values:
                        # Sort by closeness of VALUE
                        candidates = sorted(
                            [p for p in available_players if p["Name"] not in used_names],
                            key=lambda p: abs(p["Value"] - target)
                        )

                        for p in candidates:
                            if p["Name"] in used_names:
                                continue
                            if include_players and len(selected_team) < len(include_players) and p["Name"] not in include_players:
                                continue
                            if running_value_total + p["Value"] <= budget:
                                selected_team.append(p)
                                used_names.add(p["Name"])
                                running_value_total += p["Value"]
                                break

                    if len(selected_team) == team_size:
                        result_df = pd.DataFrame(selected_team)
                        st.subheader("🎯 Closest Match by Value (Greedy, Budget-Aware)")
                        st.dataframe(result_df)
                        st.write(f"**Total Value**: {round(running_value_total, 2)}")
                        st.write(f"**Total FTPS**: {round(sum(p['FTPS'] for p in selected_team), 2)}")
                        st.download_button("📥 Download Team as CSV", result_df.to_csv(index=False), file_name="closest_match_by_value.csv")
                    else:
                        st.error("❌ Couldn't build a valid team under budget using greedy value matching.")

                # Other solver modes stay the same (FTPS max, profile match) — you already have them implemented.
