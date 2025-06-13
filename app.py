import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
import re
import math
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

st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template (multi-sheet)", type=["xlsx"], key="template_upload_key"
)
available_formats, format_name = [], None
if template_file:
    try:
        xl = pd.ExcelFile(template_file)
        available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
        if available_formats:
            format_name = st.sidebar.selectbox("Select Format", available_formats)
    except Exception:
        st.sidebar.warning("⚠️ Unable to read sheets from template.")

use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)
solver_mode = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=25, value=1)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)", min_value=0, max_value=team_size, value=1
)

max_usage_pct = st.sidebar.slider(
    "Max Usage % per player/team", 0, 100, 100, 5,
    help="Cap the fraction of teams any one player can appear on (excludes forced include/exclude)."
)
ftps_rand_pct = st.sidebar.slider(
    "FTPS Randomness % for subsequent teams", 0, 100, 0, 5,
    help="Apply ± this percent random noise to FTPS values for teams 2…N."
)

st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

def load_players(f):
    try:
        df = pd.read_excel(f)
    except Exception as e:
        st.error(f"❌ Failed to read players file: {e}")
        st.stop()
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("❌ File must include 'Name' and 'Value'.")
        st.stop()
    return df

df = load_players(uploaded_file)

st.subheader("📋 Edit Player Data")
cols = ["Name", "Value"]
for c in ["Position", "FTPS", "Bracket"]:
    if c in df.columns:
        cols.append(c)
edited = st.data_editor(df[cols], use_container_width=True)

if "FTPS" not in edited.columns:
    edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]  # snapshot original FTPS

players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

bracket_fail = False
if use_bracket_constraints and "Bracket" not in edited.columns:
    st.sidebar.warning("⚠️ Bracket enabled but no 'Bracket' column present.")
    bracket_fail = True

target_values = None
if solver_mode == "Closest FTP Match" and template_file and format_name:
    try:
        prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
        raw = prof.iloc[:, 0].dropna().tolist()
        vals = [
            float(x) for x in raw
            if isinstance(x, (int, float)) or str(x).replace(".", "", 1).isdigit()
        ]
        if len(vals) < team_size:
            st.error(f"❌ Profile has fewer than {team_size} rows.")
            st.stop()
        target_values = vals[:team_size]
    except Exception as e:
        st.error(f"❌ Failed to read profile: {e}")
        st.stop()

max_usage_count = math.floor(num_teams * max_usage_pct / 100)

def add_bracket_constraints(prob, xvars):
    if use_bracket_constraints and not bracket_fail:
        groups = {}
        for p in players:
            b = p.get("Bracket")
            if b:
                groups.setdefault(b, []).append(xvars[p["Name"]])
        for grp in groups.values():
            prob += lpSum(grp) <= 1

def add_usage_constraints(prob, xvars):
    if max_usage_pct < 100:
        for p in players:
            nm = p["Name"]
            if nm in include_players:
                continue
            used = sum(1 for prev in prev_sets if nm in prev)
            prob += (used + xvars[nm] <= max_usage_count, f"MaxUsage_{nm}")

if st.sidebar.button("🚀 Optimize Teams"):
    all_teams = []
    prev_sets  = []

    # Maximize Budget Usage (unchanged) …
    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for _ in range(num_teams):
            prob = LpProblem("opt", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            cost = lpSum(x[n] * next(q["Value"] for q in players if q["Name"]==n) for n in x)
            prob += cost
            prob += lpSum(x.values()) == team_size
            prob += cost <= upper
            add_bracket_constraints(prob, x)
            add_usage_constraints(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            for prev in prev_sets:
                prob += lpSum(x[n] for n in prev) <= team_size - diff_count
            prob.solve()
            team = [p for p in players if x[p["Name"]].value()==1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
            upper = sum(p["Value"] for p in team) - 0.001

    # Maximize FTPS (re-added team_size constraint)
    elif solver_mode == "Maximize FTPS":
        for idx in range(num_teams):
            # Team 1 always raw FTPS
            if idx == 0:
                ftps_vals = {p["Name"]: p["base_FTPS"] for p in players}
            else:
                ftps_vals = {
                    p["Name"]:
                      p["base_FTPS"] * (1 + random.uniform(-ftps_rand_pct/100, ftps_rand_pct/100))
                    for p in players
                }

            prob = LpProblem("opt", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            prob += lpSum(x[n] * ftps_vals[n] for n in x)
            # re-introduced team size constraint
            prob += lpSum(x.values()) == team_size
            prob += lpSum(
                x[n] * next(q["Value"] for q in players if q["Name"]==n)
                for n in x
            ) <= budget

            add_bracket_constraints(prob, x)
            add_usage_constraints(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            for prev in prev_sets:
                prob += lpSum(x[n] for n in prev) <= team_size - diff_count

            prob.solve()
            team = [p for p in players if x[p["Name"]].value()==1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})

    # Closest FTP Match (unchanged) …
    else:
        # [Your greedy closest‐match code here]
        pass

    # Write & display …
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for i, team in enumerate(all_teams, start=1):
            df_t = pd.DataFrame(team)
            df_t["Selectie (%)"] = df_t["Name"].apply(
                lambda n: round(
                    sum(1 for t in all_teams if any(p["Name"]==n for p in t))
                    / len(all_teams)*100, 1
                )
            )
            df_t.to_excel(writer, sheet_name=f"Team{i}", index=False)
    buf.seek(0)
    for i, team in enumerate(all_teams, start=1):
        with st.expander(f"Team {i}"):
            df_t = pd.DataFrame(team)
            df_t["Selectie (%)"] = df_t["Name"].apply(
                lambda n: round(
                    sum(1 for t in all_teams if any(p["Name"]==n for p in t))
                    / len(all_teams)*100, 1
                )
            )
            st.dataframe(df_t)
    st.download_button(
        "📥 Download All Teams (Excel)", buf,
        file_name="all_teams.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
    )
