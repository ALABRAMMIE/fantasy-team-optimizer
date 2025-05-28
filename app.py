
import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
import re
from io import BytesIO
import math

st.title("Fantasy Team Optimizer")

# --- Sidebar Inputs ---
sport_options = [
    "-- Choose a sport --", "Cycling", "Speed Skating", "Formula 1", "Stock Exchange",
    "Tennis", "MotoGP", "Football", "Darts", "Cyclocross", "Golf", "Snooker",
    "Olympics", "Basketball", "Dakar Rally", "Skiing", "Rugby", "Biathlon",
    "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment", "Athletics"
]
sport = st.sidebar.selectbox("Select a sport", sport_options)

# reset state when sport changes
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for k in list(st.session_state.keys()):
        if k != "selected_sport":
            del st.session_state[k]
    st.session_state.selected_sport = sport

st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader("Upload Target Profile Template (multi-sheet)",
                                         type=["xlsx"], key="template_upload_key")

# detect format sheets matching sport
available_formats = []
format_name = None
if template_file:
    try:
        xl = pd.ExcelFile(template_file)
        available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
        if available_formats:
            format_name = st.sidebar.selectbox("Select Format", available_formats)
    except:
        st.sidebar.warning("⚠️ Unable to read sheets from template.")

use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget = st.sidebar.number_input("Max Budget", value=140.0)
# team size default from sheet name "(N)"
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)

solver_mode = st.sidebar.radio("Solver Objective", [
    "Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"
])

num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=20,
                                    value=1, step=1)
max_freq_pct = st.sidebar.number_input("Max Player Frequency (%)",
                                       min_value=0, max_value=100,
                                       value=100, step=5)
raw_allow = math.floor(max_freq_pct/100 * num_teams)
max_occurrences = raw_allow if raw_allow > 0 else (1 if max_freq_pct > 0 else 0)

# --- Player Upload & Editing ---
uploaded_file = st.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
else:
    df = pd.read_excel(uploaded_file)
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("❌ File must include 'Name' and 'Value'.")
        st.stop()

    st.subheader("📋 Edit Player Data")
    cols = ["Name", "Value"]
    if "Position" in df.columns: cols.append("Position")
    if "Rank FTPS" in df.columns: cols.append("Rank FTPS")
    if "Bracket" in df.columns: cols.append("Bracket")
    edited = st.data_editor(df[cols], use_container_width=True)

    if "Rank FTPS" in edited.columns:
        rank_map = {r: max(0, 150 - (r-1)*5) for r in range(1, 31)}
        edited["FTPS"] = edited["Rank FTPS"].apply(
            lambda x: rank_map.get(int(x), 0) if pd.notnull(x) else 0
        )
    else:
        edited["FTPS"] = 0

    bracket_fail = False
    if use_bracket_constraints and "Bracket" not in edited.columns:
        st.warning("⚠️ Bracket enabled but no 'Bracket' column present.")
        bracket_fail = True

    players = edited.to_dict("records")
    if "toggle_choices" not in st.session_state:
        st.session_state.toggle_choices = {}
    default_inc = [n for n,v in st.session_state.toggle_choices.items() if v=="✔"]
    default_exc = [n for n,v in st.session_state.toggle_choices.items() if v=="✖"]

    include_players = st.sidebar.multiselect("Players to INCLUDE",
                                             edited["Name"], default=default_inc)
    exclude_players = st.sidebar.multiselect("Players to EXCLUDE",
                                             edited["Name"], default=default_exc)

    # prepare target_values for Closest FTP Match
    target_values = None
    if solver_mode=="Closest FTP Match" and template_file and format_name:
        try:
            prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
            raw = prof.iloc[:,0].dropna().tolist()
            vals = [float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace(".","",1).isdigit()]
            if len(vals) < team_size:
                st.error(f"❌ Profile has fewer than {team_size} rows.")
            else:
                target_values = vals[:team_size]
        except Exception as e:
            st.error(f"❌ Failed to read profile: {e}")

    if st.sidebar.button("🚀 Optimize Teams"):
        frequency = {p["Name"]:0 for p in players}
        all_teams = []
        prev_teams = []
        upper_cost = budget

        def add_bracket(prob, x_vars):
            if use_bracket_constraints and not bracket_fail:
                byb = {}
                for p in players:
                    b = p.get("Bracket")
                    if b:
                        byb.setdefault(b,[]).append(x_vars[p["Name"]])
                for lst in byb.values():
                    prob += lpSum(lst) <= 1

        if solver_mode == "Maximize Budget Usage":
            # implicit minimum cost to form a team
            # implicit minimum cost to form a team, considering fixed includes
            cost_fixed = sum(p['Value'] for p in players if p['Name'] in include_players)
            remaining_slots = max(team_size - len(include_players), 0)
            available_values = sorted(p['Value'] for p in players if p['Name'] not in include_players)
            min_possible_cost = cost_fixed + sum(available_values[:remaining_slots])

            # iterative LP with decreasing upper_cost
            for _ in range(num_teams):
                if upper_cost < min_possible_cost:
                    st.info(f"🎯 Onder minimale haalbare kosten ({min_possible_cost}). Stoppen.")
                    break

                prob = LpProblem("opt", LpMaximize)
                x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
                prob += lpSum(x[n]*next(p["Value"] for p in players if p["Name"]==n) for n in x)
                prob += lpSum(x[n] for n in x) == team_size
                prob += lpSum(x[n]*next(p["Value"] for p in players if p["Name"]==n) for n in x) <= upper_cost
                add_bracket(prob, x)
                for n in include_players: prob += x[n] == 1
                for n in exclude_players: prob += x[n] == 0
                for n,c in frequency.items():
                    if n not in include_players and n not in exclude_players and c >= max_occurrences:
                        prob += x[n] == 0
                # exclude exact previous teams
                for team in prev_teams:
                    prob += lpSum(x[n] for n in team) <= team_size - 1

                prob.solve()
                if prob.status != 1:
                    st.warning(f"⚠️ LP infeasible at upper_cost={upper_cost}. Using greedy fallback.")
                    # greedy fill (budget-safe)
                    sel = []
                    while len(sel) < team_size:
                        used = {p["Name"] for p in sel}
                        current_cost = sum(p["Value"] for p in sel)
                        cands = [p for p in players if p["Name"] not in used 
                                 and current_cost + p["Value"] <= budget
                                 and (p["Name"] in include_players or p["Name"] in exclude_players 
                                      or frequency[p["Name"]] < max_occurrences)]
                        if use_bracket_constraints:
                            used_b = {q.get("Bracket") for q in sel if q.get("Bracket")}
                            cands = [p for p in cands if p.get("Bracket") not in used_b]
                        if not cands: break
                        cands.sort(key=lambda p: -p["Value"])
                        sel.append(cands[0])
                    team = sel
                else:
                    team = [p for p in players if x[p["Name"]].value() == 1]

                cost = sum(p["Value"] for p in team)
                prev_teams.append([p["Name"] for p in team])
                for p in team: frequency[p["Name"]] += 1
                all_teams.append(team)
                upper_cost = cost - 0.1

        else:
            # other solver modes...
            pass

        # compute frequency percentages
        freq_pct = {n: (c/num_teams*100) for n,c in frequency.items()}

        # write & display teams
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as wr:
            for i, team in enumerate(all_teams, start=1):
                df_t = pd.DataFrame(team)
                df_t["Selectie (%)"] = df_t["Name"].apply(lambda n: round(freq_pct.get(n,0),1))
                df_t.to_excel(wr, sheet_name=f"Team{i}", index=False)
        buf.seek(0)

        for i, team in enumerate(all_teams, start=1):
            with st.expander(f"Team {i}"):
                df_t = pd.DataFrame(team)
                df_t["Selectie (%)"] = df_t["Name"].apply(lambda n: round(freq_pct.get(n,0),1))
                st.dataframe(df_t)

        st.download_button("📥 Download All Teams (Excel)", buf,
                           file_name="all_teams.xlsx",
                           mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet")
