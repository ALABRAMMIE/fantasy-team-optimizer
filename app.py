import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

st.title("Fantasy Team Optimizer")

# Sport selection list
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

    solver_mode = st.sidebar.radio("Solver Objective", ["Maximize FTPS", "Maximize Budget Usage"])

    st.sidebar.markdown("---")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

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

            # Editable player table
            edited_df = st.data_editor(df[editable_cols], use_container_width=True)

            # ✅ FTPS calculation (on edited_df)
            if solver_mode == "Maximize FTPS" and "Rank FTPS" in edited_df.columns:
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
                players = edited_df.to_dict("records")  # ✅ Use FTPS-calculated data

                prob = LpProblem("FantasyTeam", LpMaximize)
                x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

                # ✅ Correct objective logic
                if solver_mode == "Maximize FTPS":
                    prob += lpSum(x[p["Name"]] * p["FTPS"] for p in players)
                else:
                    prob += lpSum(x[p["Name"]] * p["Value"] for p in players)

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
                if solver_mode == "Maximize FTPS":
                    st.write(f"**Total FTPS**: {sum(p['FTPS'] for p in selected)}")

                st.download_button("📥 Download Team as CSV", result_df.to_csv(index=False), file_name="optimized_team.csv")
    else:
        st.info("Please upload your Cycling Excel file to continue.")


else:
    st.warning(f"The constraint system for **{sport}** is not configured yet. Coming soon!")
