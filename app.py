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

# Reset state when sport changes
if "selected_sport" not in st.session_state or st.session_state.selected_sport != sport:
    st.session_state.clear()
    st.session_state.selected_sport = sport

st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template (multi-sheet)", type=["xlsx"], key="template_upload_key"
)

# Detect format sheets matching sport
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

# Team size input
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", min_value=1, value=default_team_size, step=1)

# Max budget dynamic default
uploaded = st.file_uploader("Upload players file next to calculate default budget...", type=["xlsx"], key="_budget_default_key")
if uploaded:
    df_test = pd.read_excel(uploaded)
    sorted_vals = sorted(df_test['Value'], reverse=True)
    budget_default = float(sum(sorted_vals[:team_size]))
else:
    budget_default = 140.0
budget = st.sidebar.number_input("Max Budget", min_value=0.0, value=budget_default, step=0.1)

solver_mode = st.sidebar.radio("Solver Objective", [
    "Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"
])

num_teams = st.sidebar.number_input(
    "Number of Teams", min_value=1, max_value=20, value=1, step=1
)
max_freq_pct = st.sidebar.number_input(
    "Max Player Frequency (%)", min_value=0, max_value=100, value=100, step=5
)
max_occurrences = math.floor(max_freq_pct / 100 * num_teams)

# --- Player Upload & Editing ---
uploaded_file = st.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

# Read and validate
df = pd.read_excel(uploaded_file)
if not {"Name", "Value"}.issubset(df.columns):
    st.error("❌ File must include 'Name' and 'Value'.")
    st.stop()

st.subheader("📋 Edit Player Data")
cols = ["Name", "Value"]
for optional in ("Position", "Rank FTPS", "Bracket"):
    if optional in df.columns:
        cols.append(optional)
edited = st.data_editor(df[cols], use_container_width=True)

# Compute FTPS
if "Rank FTPS" in edited.columns:
    rank_map = {r: max(0, 150 - (r - 1) * 5) for r in range(1, 31)}
    edited["FTPS"] = edited["Rank FTPS"].apply(lambda x: rank_map.get(int(x), 0) if pd.notnull(x) else 0)
else:
    edited["FTPS"] = 0

# Bracket check
bracket_fail = False
if use_bracket_constraints and "Bracket" not in edited.columns:
    st.warning("⚠️ Bracket enabled but no 'Bracket' column present.")
    bracket_fail = True

players = edited.to_dict("records")

# Include/Exclude toggles
if "toggle_choices" not in st.session_state:
    st.session_state.toggle_choices = {}
default_inc = [n for n,v in st.session_state.toggle_choices.items() if v=="✔"]
default_exc = [n for n,v in st.session_state.toggle_choices.items() if v=="✖"]
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"], default=default_inc)
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"], default=default_exc)

# Prepare target_values for Closest FTP Match
if solver_mode == "Closest FTP Match" and template_file and format_name:
    prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
    raw = prof.iloc[:,0].dropna().tolist()
    target_values = [float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace('.', '',1).isdigit()]
    if len(target_values) < team_size:
        st.error(f"❌ Profile has fewer than {team_size} rows.")
        st.stop()
    target_values = target_values[:team_size]
else:
    target_values = None

# Optimize button
def optimize():
    frequency = {p["Name"]: 0 for p in players}
    all_teams = []

    def add_bracket(prob, x_vars):
        if use_bracket_constraints and not bracket_fail:
            buckets = {}
            for p in players:
                b = p.get("Bracket")
                if b:
                    buckets.setdefault(b, []).append(x_vars[p["Name"]])
            for lst in buckets.values():
                prob += lpSum(lst) <= 1

    # LP solvers
    if solver_mode in ["Maximize FTPS", "Maximize Budget Usage"]:
        prev = []
        for _ in range(num_teams):
            # Primary LP
            prob = LpProblem("opt", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            if solver_mode == "Maximize FTPS":
                prob += lpSum(x[n] * next(p["FTPS"] for p in players if p["Name"] == n) for n in x)
            else:
                prob += lpSum(x[n] * next(p["Value"] for p in players if p["Name"] == n) for n in x)
            prob += lpSum(x[n] for n in x) == team_size
            prob += lpSum(x[n] * next(p["Value"] for p in players if p["Name"] == n) for n in x) <= budget
            add_bracket(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            for n,c in frequency.items():
                if n not in include_players and n not in exclude_players and c >= max_occurrences:
                    prob += x[n] == 0
            for t in prev:
                prob += lpSum(x[name] for name in t) <= team_size - 1
            prob.solve()
            status = prob.status
            sel_names = [n for n in x if x[n].value() == 1]

            # Fallback: if no feasible under budget, ignore budget and maximize usage
            if status != 1 or len(sel_names) < team_size:
                st.warning("⚠️ No feasible team under budget; finding next-best usage ignoring budget cap.")
                prob2 = LpProblem("fb", LpMaximize)
                x2 = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
                prob2 += lpSum(x2[n] * next(p["Value"] for p in players if p["Name"] == n) for n in x2)
                prob2 += lpSum(x2[n] for n in x2) == team_size
                add_bracket(prob2, x2)
                for n in include_players: prob2 += x2[n] == 1
                for n in exclude_players: prob2 += x2[n] == 0
                for n,c in frequency.items():
                    if n not in include_players and n not in exclude_players and c >= max_occurrences:
                        prob2 += x2[n] == 0
                prob2.solve()
                sel_names = [n for n in x2 if x2[n].value() == 1]
                team = [p for p in players if p["Name"] in sel_names]
            else:
                team = [p for p in players if p["Name"] in sel_names]

            prev.append([p["Name"] for p in team])
            for p in team: frequency[p["Name"]] += 1
            all_teams.append(team)

    # Greedy solver
    else:
        seen = {}
        attempts = num_teams * 100
        for _ in range(attempts):
            random.shuffle(players)
            avail = [p for p in players if p["Name"] not in exclude_players]
            sel, used, brs = [], set(), set()
            for n in include_players:
                for p in avail:
                    if p["Name"] == n:
                        sel.append(p); used.add(n); break
            if use_bracket_constraints and not bracket_fail:
                for p in avail:
                    b = p.get("Bracket")
                    if b and p["Name"] not in used and b not in brs:
                        sel.append(p); used.add(p["Name"]); brs.add(b); break
            for tgt in target_values[len(sel):]:
                cands = [p for p in avail if p["Name"] not in used and (p["Name"] in include_players or p["Name"] in exclude_players or frequency[p["Name"]] < max_occurrences)]
                if use_bracket_constraints: cands = [p for p in cands if p.get("Bracket") not in brs]
                if not cands: cands = [p for p in avail if p["Name"] not in used]
                cands.sort(key=lambda x: abs(x["Value"] - tgt))
                if not cands: break
                pick = cands[0]
                sel.append(pick); used.add(pick["Name"])
                if use_bracket_constraints: brs.add(pick.get("Bracket"))
            if len(sel) == team_size:
                err = sum((sel[i]["Value"] - target_values[i])**2 for i in range(team_size))
                key = tuple(p["Name"] for p in sel)
                if key not in seen or err < seen[key][0]:
                    seen[key] = (err, sel)
            if len(seen) >= num_teams:
                break
        for _, team in sorted(seen.values(), key=lambda x: x[0])[:num_teams]:
            for p in team: frequency[p["Name"]] += 1
            all_teams.append(team)

    # Download All Teams
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
        for i, team in enumerate(all_teams, start=1):
            pd.DataFrame(team).to_excel(wr, sheet_name=f"Team{i}", index=False)
    buf.seek(0)

    for i, team in enumerate(all_teams, start=1):
        with st.expander(f"Team {i}"):
            st.dataframe(pd.DataFrame(team))
    st.download_button("📥 Download All Teams (Excel)", buf, file_name="all_teams.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
