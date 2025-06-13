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
    "-- Choose a sport --",
    "Cycling", "Speed Skating", "Formula 1", "Stock Exchange",
    "Tennis", "MotoGP", "Football", "Darts", "Cyclocross",
    "Golf", "Snooker", "Olympics", "Basketball", "Dakar Rally",
    "Skiing", "Rugby", "Biathlon", "Handball", "Cross Country",
    "Baseball", "Ice Hockey", "American Football", "Ski Jumping",
    "MMA", "Entertainment", "Athletics"
]
sport = st.sidebar.selectbox("Select a sport", sport_options)

# Reset state when sport changes
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for k in list(st.session_state.keys()):
        if k != "selected_sport":
            del st.session_state[k]
    st.session_state.selected_sport = sport

# --- Upload Profile Template (multi-sheet) ---
st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template (multi-sheet)",
    type=["xlsx"], key="template_upload_key"
)
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

# --- Constraint Inputs ---
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
num_teams = st.sidebar.number_input(
    "Number of Teams", min_value=1, max_value=25, value=1
)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)",
    min_value=0, max_value=team_size, value=1
)

# --- New: Max usage % per player/team ---
max_usage_pct = st.sidebar.slider(
    "Max Usage % per player/team",
    0, 100, 100, 5,
    help="Cap the fraction of teams any one player can appear on (excludes forced include/exclude)."
)

# --- New: FTPS randomness % for subsequent teams ---
ftps_rand_pct = st.sidebar.slider(
    "FTPS Randomness % for subsequent teams",
    0, 100, 0, 5,
    help="Apply ± this percent random noise to FTPS values for teams 2…N."
)

# --- Upload Players File ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader(
    "Upload your Excel file (players)", type=["xlsx"]
)
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

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

# --- Edit Player Data & Snapshot base_FTPS ---
st.subheader("📋 Edit Player Data")
cols = ["Name", "Value"]
for col in ["Position", "FTPS", "Bracket"]:
    if col in df.columns:
        cols.append(col)
edited = st.data_editor(df[cols], use_container_width=True)

if "FTPS" not in edited.columns:
    edited["FTPS"] = 0
# snapshot the original
edited["base_FTPS"] = edited["FTPS"]

# rebuild players list
players = edited.to_dict("records")

# --- Include / Exclude selectors ---
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

# bracket check
bracket_fail = False
if use_bracket_constraints and "Bracket" not in edited.columns:
    st.sidebar.warning("⚠️ Bracket enabled but no 'Bracket' column present.")
    bracket_fail = True

# --- Read target profile for Closest FTP Match ---
target_values = None
if solver_mode == "Closest FTP Match" and template_file and format_name:
    try:
        prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
        raw = prof.iloc[:, 0].dropna().tolist()
        vals = [float(x) for x in raw
                if isinstance(x,(int,float))
                or str(x).replace(".", "",1).isdigit()]
        if len(vals) < team_size:
            st.error(f"❌ Profile has fewer than {team_size} rows.")
            st.stop()
        target_values = vals[:team_size]
    except Exception as e:
        st.error(f"❌ Failed to read profile: {e}")
        st.stop()

# --- Pre-calc usage cap ---
max_usage_count = math.floor(num_teams * max_usage_pct / 100)

def add_bracket_constraints(prob, x_vars):
    if use_bracket_constraints and not bracket_fail:
        groups = {}
        for p in players:
            b = p.get("Bracket")
            if b:
                groups.setdefault(b, []).append(x_vars[p["Name"]])
        for grp in groups.values():
            prob += lpSum(grp) <= 1

def add_usage_constraints(prob, x_vars):
    if max_usage_pct < 100:
        for p in players:
            nm = p["Name"]
            if nm in include_players:
                continue
            used = sum(1 for prev in prev_sets if nm in prev)
            prob += (used + x_vars[nm] <= max_usage_count, f"MaxUsage_{nm}")

# --- Optimize Teams Button ---
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams = []
    prev_sets = []

    # --- Maximize Budget Usage ---
    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for _ in range(num_teams):
            prob = LpProblem("opt", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            cost = lpSum(x[n]*next(pp["Value"] for pp in players if pp["Name"]==n)
                         for n in x)
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
            if prob.status != 1:
                st.warning(f"⚠️ Infeasible at budget ≤ {upper}.")
                st.stop()
            team = [p for p in players if x[p["Name"]].value()==1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
            upper = sum(p["Value"] for p in team) - 0.001

    # --- Maximize FTPS ---
    elif solver_mode == "Maximize FTPS":
        for idx in range(num_teams):
            # ─── Special‐case: 1 team, 0% rand, include redundant ───
            if idx==0 and num_teams==1 and ftps_rand_pct==0 and include_players:
                # solve once WITHOUT forcing include
                prob0 = LpProblem("opt0", LpMaximize)
                x0 = {p["Name"]: LpVariable(p["Name"],cat="Binary") for p in players}
                prob0 += lpSum(x0[p["Name"]]*p["base_FTPS"] for p in players)
                prob0 += lpSum(x0.values()) == team_size
                prob0 += lpSum(x0[n]*next(pp["Value"] for pp in players if pp["Name"]==n)
                                for n in x0) <= budget
                add_bracket_constraints(prob0, x0)
                add_usage_constraints(prob0, x0)
                for n in exclude_players: prob0 += x0[n] == 0
                prob0.solve()
                team0 = {n for n,v in x0.items() if v.value()==1}
                if set(include_players).issubset(team0):
                    # return that exact lineup
                    lineup = [p for p in players if p["Name"] in team0]
                    all_teams.append(lineup)
                    prev_sets.append(team0)
                    break

            # build FTPS values
            if idx == 0:
                ftps_vals = {p["Name"]: p["base_FTPS"] for p in players}
            else:
                if ftps_rand_pct > 0:
                    ftps_vals = {
                        p["Name"]:
                          p["base_FTPS"]
                          * (1 + random.uniform(-ftps_rand_pct/100, ftps_rand_pct/100))
                        for p in players
                    }
                else:
                    ftps_vals = {p["Name"]: p["base_FTPS"] for p in players}

            # solve with include/exclude
            prob = LpProblem("opt", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"],cat="Binary") for p in players}
            prob += lpSum(x[n]*ftps_vals[n] for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n]*next(pp["Value"] for pp in players if pp["Name"]==n)
                          for n in x) <= budget
            add_bracket_constraints(prob, x)
            add_usage_constraints(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            for prev in prev_sets:
                prob += lpSum(x[n] for n in prev) <= team_size - diff_count
            prob.solve()
            if prob.status != 1:
                st.warning("⚠️ LP infeasible for Maximize FTPS.")
                st.stop()
            team = [p for p in players if x[p["Name"]].value()==1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})

    # --- Closest FTP Match ---
    else:
        for _ in range(num_teams):
            slots = [None]*team_size
            used_brackets = set()
            used_names = set()
            for n in include_players:
                p0 = next(p for p in players if p["Name"]==n)
                diffs = [
                    (i, abs(p0["Value"]-target_values[i]))
                    for i in range(team_size) if slots[i] is None
                ]
                best_i = min(diffs, key=lambda x:x[1])[0]
                slots[best_i]=p0
                used_names.add(p0["Name"])
                if use_bracket_constraints and p0.get("Bracket"):
                    used_brackets.add(p0["Bracket"])
            def usage_ok(p):
                if p["Name"] in include_players: return True
                return sum(1 for prev in prev_sets if p["Name"] in prev) < max_usage_count
            for i in range(team_size):
                if slots[i] is not None: continue
                tgt = target_values[i]
                cands = [
                    p for p in players
                    if p["Name"] not in used_names
                    and p["Name"] not in exclude_players
                    and (not use_bracket_constraints or p.get("Bracket") not in used_brackets)
                    and usage_ok(p)
                ]
                if not cands: break
                pick = min(cands, key=lambda p: abs(p["Value"]-tgt))
                slots[i]=pick
                used_names.add(pick["Name"])
                if use_bracket_constraints and pick.get("Bracket"):
                    used_brackets.add(pick["Bracket"])
            cost = sum(p["Value"] for p in slots if p)
            if cost>budget:
                st.error(f"❌ Budget exceeded ({cost:.2f} > {budget:.2f}).")
                st.stop()
            names_set = {p["Name"] for p in slots if p}
            if all(len(names_set&prev) <= team_size-diff_count for prev in prev_sets):
                all_teams.append(slots)
                prev_sets.append(names_set)
                if len(all_teams)==num_teams: break

    # --- Write & display teams ---
    if not all_teams:
        st.error("❌ Geen teams gecreëerd; controleer je instellingen.")
        st.stop()

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
        "📥 Download All Teams (Excel)",
        buf,
        file_name="all_teams.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
    )
