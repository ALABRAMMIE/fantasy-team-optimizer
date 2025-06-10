import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
import re
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
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for k in list(st.session_state.keys()):
        if k != "selected_sport": del st.session_state[k]
    st.session_state.selected_sport = sport

# --- Upload Profile Template (multi-sheet) ---
st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader("Upload Target Profile Template (multi-sheet)", type=["xlsx"], key="template_upload_key")
available_formats = []
format_name = None
if template_file:
    try:
        xl = pd.ExcelFile(template_file)
        available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
        if available_formats:
            format_name = st.sidebar.selectbox("Select Format", available_formats)
    except Exception:
        st.sidebar.warning("⚠️ Unable to read sheets from template.")

# --- Constraints inputs ---
use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m: default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)
solver_mode = st.sidebar.radio("Solver Objective", ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"])
num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=25, value=1)
diff_count = st.sidebar.number_input("Min Verschil tussen Teams (aantal spelers)", min_value=0, max_value=team_size, value=1)

# --- Upload Players File ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

# Always read players from first sheet
def load_players(file):
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Failed to read players file: {e}")
        st.stop()
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("❌ File must include 'Name' and 'Value'.")
        st.stop()
    return df

df = load_players(uploaded_file)

# --- Edit player data ---
st.subheader("📋 Edit Player Data")
cols = ["Name", "Value"]
for col in ["Position", "Rank FTPS", "Bracket"]:
    if col in df.columns: cols.append(col)
edited = st.data_editor(df[cols], use_container_width=True)

# Compute FTPS
if "Rank FTPS" in edited.columns:
    rank_map = {r: max(0, 150 - (r-1)*5) for r in range(1, 31)}
    edited["FTPS"] = edited["Rank FTPS"].apply(lambda x: rank_map.get(int(x), 0) if pd.notnull(x) else 0)
else:
    edited["FTPS"] = 0

# Bracket check
bracket_fail = False
if use_bracket_constraints and "Bracket" not in edited.columns:
    st.warning("⚠️ Bracket enabled but no 'Bracket' column present.")
    bracket_fail = True

players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

# --- Read target profile values for Closest FTP Match ---
target_values = None
if solver_mode == "Closest FTP Match" and template_file and format_name:
    try:
        prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
        raw = prof.iloc[:, 0].dropna().tolist()
        vals = [float(x) for x in raw if isinstance(x, (int, float)) or str(x).replace(".", "", 1).isdigit()]
        if len(vals) < team_size:
            st.error(f"❌ Profile has fewer than {team_size} rows.")
            st.stop()
        target_values = vals[:team_size]
    except Exception as e:
        st.error(f"❌ Failed to read profile: {e}")
        st.stop()

# --- Optimize ---
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams = []
    prev_sets = []

    def add_bracket_constraints(prob, x_vars):
        if use_bracket_constraints and not bracket_fail:
            groups = {}
            for p in players:
                b = p.get("Bracket")
                if b:
                    groups.setdefault(b, []).append(x_vars[p["Name"]])
            for vars_list in groups.values():
                prob += lpSum(vars_list) <= 1

    # Maximize Budget Usage
    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for _ in range(num_teams):
            prob = LpProblem("opt", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            cost_expr = lpSum(x[n] * next(p["Value"] for p in players if p["Name"] == n) for n in x)
            prob += cost_expr
            prob += lpSum(x.values()) == team_size
            prob += cost_expr <= upper
            add_bracket_constraints(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            for prev in prev_sets: prob += lpSum(x[n] for n in prev) <= team_size - diff_count
            prob.solve()
            if prob.status != 1:
                st.warning(f"⚠️ Infeasible at budget <= {upper}.")
                st.stop()
            team = [p for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
            upper = sum(p["Value"] for p in team) - 0.001

    # Maximize FTPS
    elif solver_mode == "Maximize FTPS":
        for _ in range(num_teams):
            prob = LpProblem("opt", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            ftps_expr = lpSum(x[n] * next(p["FTPS"] for p in players if p["Name"] == n) for n in x)
            prob += ftps_expr
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * next(p["Value"] for p in players if p["Name"] == n) for n in x) <= budget
            add_bracket_constraints(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            for prev in prev_sets: prob += lpSum(x[n] for n in prev) <= team_size - diff_count
            prob.solve()
            if prob.status != 1:
                st.warning("⚠️ LP infeasible for Maximize FTPS.")
                st.stop()
            team = [p for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})

    # Closest FTP Match uses exact-match logic
    else:
        for _ in range(num_teams):
            sel = []
            used_brackets = set()
            # Force includes
            for n in include_players:
                p0 = next(p for p in players if p["Name"] == n)
                sel.append(p0)
                if use_bracket_constraints:
                    used_brackets.add(p0.get("Bracket"))
            # Add one from each bracket if needed
            if use_bracket_constraints:
                for p in players:
                    b = p.get("Bracket")
                    if b and p["Name"] not in [q["Name"] for q in sel] and b not in used_brackets:
                        sel.append(p)
                        used_brackets.add(b)
            # Greedy closest-match for remaining slots
            for idx in range(len(sel), team_size):
                tgt = target_values[idx]
                cands = [p for p in players if p["Name"] not in [q["Name"] for q in sel] and p["Name"] not in exclude_players]
                if use_bracket_constraints:
                    cands = [p for p in cands if p.get("Bracket") not in used_brackets]
                if not cands:
                    break
                # exact closest: INDEX/MATCH equivalent
                pick = min(cands, key=lambda p: abs(p["Value"] - tgt))
                sel.append(pick)
                if use_bracket_constraints:
                    used_brackets.add(pick.get("Bracket"))
            if len(sel) == team_size and all(len({p["Name"] for p in sel} & prev) <= team_size - diff_count for prev in prev_sets):
                all_teams.append(sel)
                prev_sets.append({p["Name"] for p in sel})
                if len(all_teams) == num_teams:
                    break

    # No teams guard
    if not all_teams:
        st.error("❌ Geen teams gecreëerd; controleer je instellingen of probeer andere parameters.")
        st.stop()

    # Write & display teams
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for i, team in enumerate(all_teams, start=1):
            df_t = pd.DataFrame(team)
            df_t["Selectie (%)"] = df_t["Name"].apply(lambda n: round(sum(1 for t in all_teams if any(p["Name"] == n for p in t)) / len(all_teams) * 100, 1))
            df_t.to_excel(writer, sheet_name=f"Team{i}", index=False)
    buf.seek(0)

    for i, team in enumerate(all_teams, start=1):
        with st.expander(f"Team {i}"):
            df_t = pd.DataFrame(team)
            df_t["Selectie (%)"] = df_t["Name"].apply(lambda n: round(sum(1 for t in all_teams if any(p["Name"] == n for p in t)) / len(all_teams) * 100, 1))
            st.dataframe(df_t)

    st.download_button("📥 Download All Teams (Excel)", buf, file_name="all_teams.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
