import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random, re, math
from io import BytesIO

st.title("Fantasy Team Optimizer")

# --- Sidebar: select sport & template ---
sport_options = [
    "-- Choose a sport --","Cycling","Speed Skating","Formula 1","Stock Exchange",
    "Tennis","MotoGP","Football","Darts","Cyclocross","Golf","Snooker",
    "Olympics","Basketball","Dakar Rally","Skiing","Rugby","Biathlon",
    "Handball","Cross Country","Baseball","Ice Hockey","American Football",
    "Ski Jumping","MMA","Entertainment","Athletics"
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
        st.sidebar.warning("⚠️ Unable to read sheets from template.")

use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget         = st.sidebar.number_input("Max Budget", value=140.0)
default_ts     = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_ts = int(m.group(1))
team_size      = st.sidebar.number_input("Team Size", min_value=1, value=default_ts, step=1)
solver_mode    = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams      = st.sidebar.number_input("Number of Teams", 1, 25, 1)
diff_count     = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)", 0, team_size, 1
)

global_max_usage = st.sidebar.slider(
    "Default Max Usage % (no bracket)", 0,100,100,5,
    help="Max usage % for players without a bracket."
)

# --- players upload & edit ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue."); st.stop()

df = pd.read_excel(uploaded_file)
if not {"Name","Value"}.issubset(df.columns):
    st.error("❌ File must include 'Name' and 'Value'."); st.stop()

st.subheader("📋 Edit Player Data")
cols   = ["Name","Value"] + [c for c in ("Position","FTPS","Bracket") if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True)
if "FTPS" not in edited:
    edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]

players         = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

bracket_fail = False
if use_bracket_constraints and "Bracket" not in edited.columns:
    st.sidebar.warning("⚠️ No 'Bracket' column but Bracket Constraints is on.")
    bracket_fail = True

# --- per-bracket slider groups in expanders ---
brackets = sorted(edited["Bracket"].dropna().unique())
bracket_rand  = {}
bracket_usage = {}
bracket_min   = {}
if brackets:
    with st.sidebar.expander("FTPS Randomness by Bracket", expanded=False):
        for b in brackets:
            bracket_rand[b] = st.slider(
                f"Bracket {b} FTPS Random %", 0,100,0,5, key=f"rand_{b}"
            )
    with st.sidebar.expander("Usage % by Bracket", expanded=False):
        for b in brackets:
            bracket_usage[b] = st.slider(
                f"Bracket {b} Max Usage %", 0,100,100,5, key=f"use_{b}"
            )
            bracket_min[b] = st.slider(
                f"Bracket {b} Min Usage %", 0,100,0,5, key=f"minuse_{b}"
            )

# --- target profile for Closest FTP Match ---
target_values = None
if solver_mode == "Closest FTP Match" and template_file and format_name:
    prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
    raw  = prof.iloc[:,0].dropna().tolist()
    vals = [float(x) for x in raw
            if isinstance(x,(int,float)) or str(x).replace(".", "",1).isdigit()]
    if len(vals) < team_size:
        st.error(f"❌ Profile has fewer than {team_size} rows."); st.stop()
    target_values = vals[:team_size]

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
    # skip for Closest FTP Match or single-team
    if solver_mode == "Closest FTP Match" or num_teams <= 1:
        return
    for p in players:
        nm = p["Name"]
        if nm in include_players:
            continue
        pct_max = bracket_usage.get(p.get("Bracket"), global_max_usage)
        pct_min = bracket_min.get(p.get("Bracket"), 0)
        cap_max = math.floor(num_teams * pct_max / 100)
        cap_min = math.ceil (num_teams * pct_min / 100)
        used    = sum(1 for prev in prev_sets if nm in prev)
        # enforce min and max usage across teams
        prob += (used + xvars[nm] <= cap_max, f"MaxUsage_{nm}")
        prob += (used + xvars[nm] >= cap_min, f"MinUsage_{nm}")

if st.sidebar.button("🚀 Optimize Teams"):
    all_teams = []
    prev_sets  = []

    # --- Maximize Budget Usage ---
    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for _ in range(num_teams):
            prob = LpProblem("opt", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            cost = lpSum(x[n] * next(q["Value"] for q in players if q["Name"] == n) for n in x)
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
            team = [{**p, "Adjusted FTPS": p["base_FTPS"]}
                    for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
            upper = sum(p["Value"] for p in team) - 0.001

    # --- Maximize FTPS ---
    elif solver_mode == "Maximize FTPS":
        for idx in range(num_teams):
            if idx == 0:
                ftps_vals = {p["Name"]: p["base_FTPS"] for p in players}
            else:
                ftps_vals = {}
                for p in players:
                    r = bracket_rand.get(p.get("Bracket"), 0)
                    ftps_vals[p["Name"]] = p["base_FTPS"] * (1 + random.uniform(-r/100, r/100))

            prob = LpProblem("opt", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
            prob += lpSum(x[n] * ftps_vals[n] for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * next(q["Value"] for q in players if q["Name"] == n) for n in x) <= budget
            add_bracket_constraints(prob, x)
            add_usage_constraints(prob, x)
            for n in include_players: prob += x[n] == 1
            for n in exclude_players: prob += x[n] == 0
            for prev in prev_sets:
                prob += lpSum(x[n] for n in prev) <= team_size - diff_count
            prob.solve()
            team = [{**p, "Adjusted FTPS": ftps_vals[p["Name"]]}
                    for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})

    # --- Closest FTP Match (unaffected) ---
    else:
        for _ in range(num_teams):
            slots, used_brackets, used_names = [None]*team_size, set(), set()
            # place includes
            for n in include_players:
                p0 = next(p for p in players if p["Name"] == n)
                diffs = [(i, abs(p0["Value"] - target_values[i])) for i in range(team_size) if slots[i] is None]
                best_i = min(diffs, key=lambda x: x[1])[0]
                slots[best_i] = p0
                used_names.add(n)
                if use_bracket_constraints and p0.get("Bracket"):
                    used_brackets.add(p0["Bracket"])
            # greedy fill
            for i in range(team_size):
                if slots[i] is not None: continue
                tgt = target_values[i]
                cands = [
                    p for p in players
                    if p["Name"] not in used_names
                    and p["Name"] not in exclude_players
                    and (not use_bracket_constraints or p.get("Bracket") not in used_brackets)
                ]
                if not cands: break
                pick = min(cands, key=lambda p: abs(p["Value"] - tgt))
                slots[i] = pick
                used_names.add(pick["Name"])
                if use_bracket_constraints and pick.get("Bracket"):
                    used_brackets.add(pick["Bracket"])
            cost = sum(p["Value"] for p in slots if p)
            if cost > budget:
                st.error(f"❌ Budget exceeded ({cost:.2f} > {budget:.2f})."); st.stop()
            names_set = {p["Name"] for p in slots if p}
            if all(len(names_set & prev) <= team_size - diff_count for prev in prev_sets):
                team = [{**p, "Adjusted FTPS": p["base_FTPS"]} for p in slots if p]
                all_teams.append(team)
                prev_sets.append(names_set)
                if len(all_teams) == num_teams:
                    break

    # --- Write & display teams ---
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
