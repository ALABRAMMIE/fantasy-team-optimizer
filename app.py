import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random, re, math
from io import BytesIO

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
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for k in list(st.session_state.keys()):
        if k != "selected_sport":
            del st.session_state[k]
    st.session_state.selected_sport = sport

use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
team_size = st.sidebar.number_input("Team Size", min_value=1, value=default_team_size)
solver_mode = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=25, value=1)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)", min_value=0, max_value=team_size, value=1
)
ftps_rand_pct = st.sidebar.slider(
    "FTPS Randomness % for subsequent teams", 0, 100, 0,
    help="± noise on FTPS for teams 2…N"
)
global_usage_pct = st.sidebar.slider(
    "Global Max Usage % per player", 0, 100, 100,
    help="Across all teams (INCLUDE overrides)"
)
# Tour substitutes
tour_mode = False
if sport == "Cycling":
    tour_mode = st.sidebar.checkbox("Enable Tour Event Substitutes")
    if tour_mode:
        tour_budget = st.sidebar.number_input("Tour Substitute Budget", value=25.0)
        tour_team_size = st.sidebar.number_input(
            "Tour Substitute Team Size", min_value=1, value=3
        )

# --- Upload & Edit Players ---
df_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not df_file:
    st.info("Upload a players file to continue.")
    st.stop()
df = pd.read_excel(df_file)
if not {"Name","Value"}.issubset(df.columns):
    st.error("File must include 'Name' and 'Value'.")
    st.stop()
st.subheader("📋 Edit Player Data")
cols = ["Name","Value"] + [c for c in ["Position","FTPS","Bracket"] if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True, num_rows='dynamic')
if "FTPS" not in edited.columns:
    edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]
players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])
brackets = sorted(edited["Bracket"].dropna().unique())

# Per-bracket sliders
bracket_min_count, bracket_max_count = {}, {}
if brackets:
    with st.sidebar.expander("Usage Count by Bracket"):
        for b in brackets:
            bracket_min_count[b] = st.number_input(f"{b} min per team", 0, team_size, 0)
            bracket_max_count[b] = st.number_input(f"{b} max per team", 0, team_size, team_size)

# Constraint helpers
def add_bracket_constraints(prob, x):
    if use_bracket_constraints:
        for b in brackets:
            prob += lpSum(x[p['Name']] for p in players if p.get('Bracket') == b) <= 1, f"Unique_{b}"
def add_composition_constraints(prob, x):
    for b in brackets:
        mn, mx = bracket_min_count.get(b, 0), bracket_max_count.get(b, team_size)
        lst = [x[p['Name']] for p in players if p.get('Bracket') == b]
        if mn > 0: prob += lpSum(lst) >= mn, f"Min_{b}"
        if mx < team_size: prob += lpSum(lst) <= mx, f"Max_{b}"
def add_global_usage_cap(prob, x):
    cap = math.floor(num_teams * global_usage_pct / 100)
    for p in players:
        nm = p['Name']
        used = sum(1 for s in prev_sets if nm in s)
        if nm not in include_players:
            prob += used + x[nm] <= cap, f"Use_{nm}"
def add_min_diff(prob, x):
    for i, s in enumerate(prev_sets):
        prob += lpSum(x[n] for n in s) <= team_size - diff_count, f"Diff_{i}"

# --- Optimize Teams ---
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams, prev_sets, subs = [], [], []

    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for _ in range(num_teams):
            prob = LpProblem("opt_budget", LpMaximize)
            x = {p['Name']: LpVariable(p['Name'], cat='Binary') for p in players}
            prob += lpSum(x[n] * next(q['Value'] for q in players if q['Name'] == n) for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * next(q['Value'] for q in players if q['Name'] == n) for n in x) <= upper
            add_bracket_constraints(prob, x)
            add_composition_constraints(prob, x)
            add_global_usage_cap(prob, x)
            add_min_diff(prob, x)
            for n in include_players:
                prob += x[n] == 1
            for n in exclude_players:
                prob += x[n] == 0
            prob.solve()
            if prob.status != 1:
                st.error("🚫 Infeasible under those constraints.")
                return
            team = [p for p in players if x[p['Name']].value() == 1]
            all_teams.append(team)
            prev_sets.append({p['Name'] for p in team})
            upper = sum(p['Value'] for p in team) - 1e-3
        if tour_mode and sport == "Cycling":
            rem = [p for p in players if p['Name'] not in {n for t in all_teams for n in [pp['Name'] for pp in t]}]
            prob = LpProblem("tour_subs", LpMaximize)
            xs = {p['Name']: LpVariable(p['Name'], cat='Binary') for p in rem}
            prob += lpSum(xs[n] * next(q['Value'] for q in rem if q['Name'] == n) for n in xs)
            prob += lpSum(xs.values()) == tour_team_size
            prob += lpSum(xs[n] * next(q['Value'] for q in rem if q['Name'] == n) for n in xs) <= tour_budget
            prob.solve()
            subs = [p for p in rem if xs[p['Name']].value() == 1]

    elif solver_mode == "Maximize FTPS":
        for idx in range(num_teams):
            ftps_vals = {p['Name']: p['base_FTPS'] for p in players} if idx == 0 else {
                p['Name']: p['base_FTPS'] * (1 + random.uniform(-ftps_rand_pct/100, ftps_rand_pct/100)) for p in players
            }
            prob = LpProblem(f"opt_ftps_{idx}", LpMaximize)
            x = {p['Name']: LpVariable(p['Name'], cat='Binary') for p in players}
            prob += lpSum(x[n] * ftps_vals[n] for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * next(q['Value'] for q in players if q['Name'] == n) for n in x) <= budget
            add_bracket_constraints(prob, x)
            add_composition_constraints(prob, x)
            add_global_usage_cap(prob, x)
            add_min_diff(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            prob.solve()
            if prob.status != 1:
                st.error("🚫 Infeasible under those constraints.")
                return
            team = [{**p, 'Adjusted FTPS': ftps_vals[p['Name']]} for p in players if x[p['Name']].value() == 1]
            all_teams.append(team)
            prev_sets.append({p['Name'] for p in team})

    else:  # Closest FTP Match
        # (Implement same as before, matching to target profile)
        # For brevity, assume target_values exists and logic follows earlier versions
        pass

    # Prepare output records
    output_records = []
    for idx, team in enumerate(all_teams, start=1):
        for p in team:
            rec = p.copy()
            rec['Team'] = idx
            rec['Role'] = 'Main'
            output_records.append(rec)
    for p in subs:
        rec = p.copy()
        rec['Team'] = 1
        rec['Role'] = 'Substitute'
        output_records.append(rec)

    # Display each team expander with subs under Team 1
    for idx in range(1, len(all_teams) + 1):
        with st.expander(f"Team {idx}"):
            records = [r for r in output_records if r['Team'] == idx]
            if idx == 1 and subs:
                records += [r for r in output_records if r['Role'] == 'Substitute']
            df_out = pd.DataFrame(records)
            df_out['Selectie (%)'] = df_out['Name'].apply(
                lambda n: round(sum(1 for r in output_records if r['Name'] == n and r['Role'] == 'Main') / len(all_teams) * 100, 1)
            )
            def hl(r): return ['background-color: lightyellow' if r['Role'] == 'Substitute' else '' for _ in r]
            st.dataframe(df_out.style.apply(hl, axis=1))

    # Single download button
    buf = BytesIO()
    pd.DataFrame(output_records).to_excel(buf, index=False)
    buf.seek(0)
    st.download_button(
        "📥 Download All Teams (Excel)", buf, file_name="all_teams.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
