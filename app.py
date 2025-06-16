import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random, re, math
from io import BytesIO

st.title("Fantasy Team Optimizer")

# --- Sidebar: Sport & Template ---
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
    except:
        st.sidebar.warning("⚠️ Unable to read template.")

# --- Core constraints ---
use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget            = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input(
    "Team Size", min_value=1, value=default_team_size, step=1
)
solver_mode = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams  = st.sidebar.number_input("Number of Teams", min_value=1, max_value=25, value=1)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)", min_value=0, max_value=team_size, value=1
)

# --- FTPS randomness ---
ftps_rand_pct = st.sidebar.slider(
    "FTPS Randomness % for subsequent teams", 0, 100, 0, 5,
    help="± this percent noise on FTPS for teams 2…N"
)

# --- Upload & Edit Players ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Failed to read players file: {e}")
    st.stop()

if not {"Name", "Value"}.issubset(df.columns):
    st.error("❌ File must include 'Name' and 'Value'.")
    st.stop()

st.subheader("📋 Edit Player Data")
cols = ["Name", "Value"] + [c for c in ("Position", "FTPS", "Bracket") if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True)
if "FTPS" not in edited.columns:
    edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]

players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

# collect brackets
brackets = sorted(edited["Bracket"].dropna().unique())
if use_bracket_constraints and not brackets:
    st.sidebar.warning("⚠️ Bracket Constraints on but no ‘Bracket’ column found.")

# --- Per-Bracket Min/Max count sliders ---
bracket_min_count = {}
bracket_max_count = {}
if brackets:
    with st.sidebar.expander("Usage Count by Bracket", expanded=False):
        for b in brackets:
            bracket_min_count[b] = st.number_input(
                f"Bracket {b} Min picks per team", 0, team_size, 0, 1, key=f"min_{b}"
            )
            bracket_max_count[b] = st.number_input(
                f"Bracket {b} Max picks per team", 0, team_size, team_size, 1, key=f"max_{b}"
            )

# --- Read target profile for Closest FTP Match ---
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

# --- Constraint helpers ---
def add_bracket_constraints(prob, x):
    if use_bracket_constraints:
        for b in brackets:
            members = [x[p["Name"]] for p in players if p.get("Bracket") == b]
            prob += lpSum(members) <= 1, f"UniqueBracket_{b}"

def add_composition_constraints(prob, x):
    # per-team bracket pick counts
    for b in brackets:
        mn = bracket_min_count.get(b, 0)
        mx = bracket_max_count.get(b, team_size)
        members = [x[p["Name"]] for p in players if p.get("Bracket") == b]
        if mn > 0:
            prob += lpSum(members) >= mn, f"MinBracket_{b}"
        if mx < team_size:
            prob += lpSum(members) <= mx, f"MaxBracket_{b}"

def add_min_diff(prob, x):
    # enforce difference from each previous team
    for idx, prev in enumerate(prev_sets):
        prob += lpSum(x[n] for n in prev) <= team_size - diff_count, f"MinDiff_{idx}"

# --- Optimize Teams ---
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams = []
    prev_sets  = []

    # --- Maximize Budget Usage ---
    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for _ in range(num_teams):
            prob = LpProblem("opt_budget", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            # objective & core constraints
            prob += lpSum(
                x[n] * next(q["Value"] for q in players if q["Name"] == n)
                for n in x
            )
            prob += lpSum(x.values()) == team_size
            prob += lpSum(
                x[n] * next(q["Value"] for q in players if q["Name"] == n)
                for n in x
            ) <= upper

            add_bracket_constraints(prob, x)
            add_composition_constraints(prob, x)
            add_min_diff(prob, x)

            for n in include_players:
                prob += x[n] == 1
            for n in exclude_players:
                prob += x[n] == 0

            prob.solve()
            if prob.status != 1:
                st.error("🚫 Infeasible under those constraints.")
                st.stop()

            team = [p for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
            upper = sum(p["Value"] for p in team) - 0.001

    # --- Maximize FTPS ---
    elif solver_mode == "Maximize FTPS":
        for idx in range(num_teams):
            # FTPS values: raw for team1, noise for 2…N
            if idx == 0:
                ftps_vals = {p["Name"]: p["base_FTPS"] for p in players}
            else:
                ftps_vals = {
                    p["Name"]: p["base_FTPS"] * (1 + random.uniform(
                        -ftps_rand_pct/100, ftps_rand_pct/100
                    ))
                    for p in players
                }

            prob = LpProblem(f"opt_ftps_{idx}", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            prob += lpSum(x[n] * ftps_vals[n] for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(
                x[n] * next(q["Value"] for q in players if q["Name"] == n)
                for n in x
            ) <= budget

            add_bracket_constraints(prob, x)
            add_composition_constraints(prob, x)
            add_min_diff(prob, x)

            for n in include_players:
                prob += x[n] == 1
            for n in exclude_players:
                prob += x[n] == 0

            prob.solve()
            if prob.status != 1:
                st.error("🚫 Infeasible under those constraints.")
                st.stop()

            team = [
                {**p, "Adjusted FTPS": ftps_vals[p["Name"]]}
                for p in players if x[p["Name"]].value() == 1
            ]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})

    # --- Closest FTP Match ---
    else:
        for _ in range(num_teams):
            slots, used_brackets, used_names = [None]*team_size, set(), set()

            # place includes
            for n in include_players:
                p0 = next(p for p in players if p["Name"] == n)
                diffs = [
                    (i, abs(p0["Value"] - target_values[i]))
                    for i in range(team_size) if slots[i] is None
                ]
                best_i = min(diffs, key=lambda x: x[1])[0]
                slots[best_i] = p0
                used_names.add(n)
                if use_bracket_constraints and p0.get("Bracket"):
                    used_brackets.add(p0["Bracket"])

            # greedy fill
            for i in range(team_size):
                if slots[i] is not None:
                    continue
                tgt = target_values[i]
                cands = [
                    p for p in players
                    if p["Name"] not in used_names
                    and p["Name"] not in exclude_players
                    and (not use_bracket_constraints or p.get("Bracket") not in used_brackets)
                ]
                if not cands:
                    break
                pick = min(cands, key=lambda p: abs(p["Value"] - tgt))
                slots[i] = pick
                used_names.add(pick["Name"])
                if use_bracket_constraints and pick.get("Bracket"):
                    used_brackets.add(pick["Bracket"])

            cost = sum(p["Value"] for p in slots if p)
            if cost > budget:
                st.error(f"❌ Budget exceeded ({cost:.2f} > {budget:.2f}).")
                st.stop()

            current = {p["Name"] for p in slots if p}
            if prev_sets and len(current & prev_sets[-1]) > team_size - diff_count:
                st.error("🚫 Violation of min-difference constraint.")
                st.stop()

            team = [{**p, "Adjusted FTPS": p["base_FTPS"]} for p in slots if p]
            all_teams.append(team)
            prev_sets.append(current)
            if len(all_teams) == num_teams:
                break

    # --- Output & Download ---
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for i, team in enumerate(all_teams, start=1):
            df_t = pd.DataFrame(team)
            df_t["Selectie (%)"] = df_t["Name"].apply(
                lambda n: round(
                    sum(1 for t in all_teams if any(p["Name"] == n for p in t))
                    / len(all_teams) * 100, 1
                )
            )
            df_t.to_excel(writer, sheet_name=f"Team{i}", index=False)
    buf.seek(0)

    for i, team in enumerate(all_teams, start=1):
        with st.expander(f"Team {i}"):
            df_t = pd.DataFrame(team)
            df_t["Selectie (%)"] = df_t["Name"].apply(
                lambda n: round(
                    sum(1 for t in all_teams if any(p["Name"] == n for p in t))
                    / len(all_teams) * 100, 1
                )
            )
            st.dataframe(df_t)

    st.download_button(
        "📥 Download All Teams (Excel)",
        buf,
        file_name="all_teams.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
    )
