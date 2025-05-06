import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

st.title("Fantasy Team Optimizer")

# Sport selection dropdown
sport = st.sidebar.selectbox("Kies een sport (Choose a sport)", [
    "WIELRENNEN", "SCHAATSEN", "FORMULA 1", "STOCK EXCHANGE", "TENNIS", "MOTOGP", "FOOTBALL",
    "DARTS", "CYCLOCROSS", "GOLF", "SNOOKER", "OLYMPISCHE SPELEN", "BASKETBALL", "DAKAR",
    "SKIING", "RUGBY", "BIATHLON", "HANDBALL", "CROSS COUNTRY", "BASEBALL", "ICE HOCKEY",
    "AMERICAN FOOTBALL", "SCHANSSPRINGEN", "MMA", "ENTERTAINMENT"
])

uploaded_file = st.file_uploader("Upload je Excel bestand (Upload your Excel file)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader(f"📄 Geüploade gegevens ({sport})")
    st.dataframe(df)

    st.sidebar.header("🔧 Instellingen")
    budget = st.sidebar.number_input("Max Budget", value=140.0)
    team_size = st.sidebar.number_input("Aantal Spelers", value=13, step=1)

    include_players = st.sidebar.multiselect("Spelers verplicht toevoegen", df["Name"])
    exclude_players = st.sidebar.multiselect("Spelers uitsluiten", df["Name"])

    if st.button("Optimaliseer Team"):
        players = df.to_dict("records")

        prob = LpProblem("FantasyTeam", LpMaximize)

        # Binary decision variables
        x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

        # Objective: Maximize FTPS
        prob += lpSum(x[p["Name"]] * p["FTPS"] for p in players)

        # Constraints
        prob += lpSum(x[p["Name"]] * p["Value"] for p in players) <= budget
        prob += lpSum(x[p["Name"]] for p in players) == team_size

        for name in include_players:
            prob += x[name] == 1

        for name in exclude_players:
            prob += x[name] == 0

        # Solve the problem
        prob.solve()

        selected = [p for p in players if x[p["Name"]].value() == 1]

        st.subheader("✅ Geoptimaliseerd Team")
        result_df = pd.DataFrame(selected)
        st.dataframe(result_df)

        total_value = sum(p["Value"] for p in selected)
        total_ftps = sum(p["FTPS"] for p in selected)

        st.write(f"**Totale Waarde**: {total_value}")
        st.write(f"**Totale FTPS**: {total_ftps}")

        st.download_button("📥 Download Team als CSV", result_df.to_csv(index=False), file_name="optimized_team.csv")

