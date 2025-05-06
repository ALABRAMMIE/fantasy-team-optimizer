import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum

st.title("Fantasy Team Optimizer")

# Sport selection: Dutch with English in parentheses
sport = st.sidebar.selectbox("Kies een sport (Choose a sport)", [
    "WIELRENNEN (Cycling)",
    "SCHAATSEN (Speed Skating)",
    "FORMULA 1 (Formula 1)",
    "STOCK EXCHANGE (Stock Exchange)",
    "TENNIS (Tennis)",
    "MOTOGP (MotoGP)",
    "FOOTBALL (Football)",
    "DARTS (Darts)",
    "CYCLOCROSS (Cyclocross)",
    "GOLF (Golf)",
    "SNOOKER (Snooker)",
    "OLYMPISCHE SPELEN (Olympics)",
    "BASKETBALL (Basketball)",
    "DAKAR (Dakar Rally)",
    "SKIING (Skiing)",
    "RUGBY (Rugby)",
    "BIATHLON (Biathlon)",
    "HANDBALL (Handball)",
    "CROSS COUNTRY (Cross Country)",
    "BASEBALL (Baseball)",
    "ICE HOCKEY (Ice Hockey)",
    "AMERICAN FOOTBALL (American Football)",
    "SCHANSSPRINGEN (Ski Jumping)",
    "MMA (MMA)",
    "ENTERTAINMENT (Entertainment)"
])

uploaded_file = st.file_uploader("Upload je Excel bestand (Upload your Excel file)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader(f"📄 Geüploade gegevens voor: {sport}")
    st.dataframe(df)

    st.sidebar.header("🔧 Instellingen (Settings)")
    budget = st.sidebar.number_input("Max Budget", value=140.0)
    team_size = st.sidebar.number_input("Aantal Spelers (Team Size)", value=13, step=1)

    include_players = st.sidebar.multiselect("Spelers verplicht toevoegen (Include)", df["Name"])
    exclude_players = st.sidebar.multiselect("Spelers uitsluiten (Exclude)", df["Name"])

    if st.button("Optimaliseer Team (Optimize Team)"):
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

        st.write(f"**Totale Waarde (Total Value)**: {total_value}")
        st.write(f"**Totale FTPS**: {total_ftps}")

        st.download_button("📥 Download Team als CSV", result_df.to_csv(index=False), file_name="optimized_team.csv")


