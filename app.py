import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

st.title("Fantasy Team Optimizer")

# Select sport
sports = ["-- Choose a sport --", "Cycling", "Football", "Tennis"]
sport = st.sidebar.selectbox("Select a sport", sports)

if sport == "Cycling":
    st.sidebar.header("🚴 Cycling Constraints")
    budget = st.sidebar.number_input("Max Budget", value=140.0)
    team_size = st.sidebar.number_input("Team Size", value=13, step=1)

    uploaded_file = st.file_uploader("Upload your Excel file (with Expected Rank column)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        if "Expected Rank" not in df.columns:
            st.error("Missing 'Expected Rank' column in your Excel file.")
        else:
            st.subheader("📄 Uploaded Data")
            st.dataframe(df)

            # Default Top 30 rank-to-points table
            default_points = {rank: max(0, 150 - (rank - 1) * 5) for rank in range(1, 31)}
            rank_df = pd.DataFrame({
                "Rank": list(default_points.keys()),
                "Points": list(default_points.values())
            })

            st.subheader("🎯 Rank to FTPS Points Mapping")
            edited_rank_df = st.data_editor(rank_df, use_container_width=True, num_rows="fixed")

            # Build mapping from the edited table
            rank_points = dict(zip(edited_rank_df["Rank"], edited_rank_df["Points"]))

            if st.button("Calculate FTPS from Rank"):
                df["FTPS"] = df["Expected Rank"].apply(lambda r: rank_points.get(int(r), 0))
                st.success("FTPS calculated!")
                st.dataframe(df)

                include_players = st.sidebar.multiselect("Players to INCLUDE", df["Name"])
                exclude_players = st.sidebar.multiselect("Players to EXCLUDE", df["Name"])

                if st.button("Optimize Team"):
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

                    st.subheader("✅ Optimized Team")
                    result_df = pd.DataFrame(selected)
                    st.dataframe(result_df)

                    st.write(f"**Total Value**: {sum(p['Value'] for p in selected)}")
                    st.write(f"**Total FTPS**: {sum(p['FTPS'] for p in selected)}")

                    st.download_button("📥 Download Team as CSV", result_df.to_csv(index=False), file_name="optimized_team.csv")
else:
    st.info("Choose 'Cycling' to get started. Other sports coming soon.")
