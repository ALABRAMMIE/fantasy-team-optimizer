import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
import re
import math
from io import BytesIO

st.title("Fantasy Team Optimizer")

# --- Sidebar: select sport & template ---
sport_options = [
    "-- Choose a sport --", "Cycling","Speed Skating","Formula 1","Stock Exchange",
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
        if k!="selected_sport": del st.session_state[k]
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

# --- Constraints inputs ---
use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget     = st.sidebar.number_input("Max Budget", value=140.0)
default_ts = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m: default_ts = int(m.group(1))
team_size   = st.sidebar.number_input("Team Size", default_ts, step=1)
solver_mode = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS","Maximize Budget Usage","Closest FTP Match"]
)
num_teams  = st.sidebar.number_input("Number of Teams", 1, 25, 1)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)", 0, team_size, 1
)

# --- Global fallback usage slider (for players w/o bracket) ---
global_max_usage = st.sidebar.slider(
    "Default Max Usage % (no bracket)",
    0,100,100,5,
    help="Max usage % for any player without a bracket."
)

# --- Upload & edit players ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue."); st.stop()

df = pd.read_excel(uploaded_file)
if not {"Name","Value"}.issubset(df.columns):
    st.error("❌ File must include 'Name' and 'Value'."); st.stop()

st.subheader("📋 Edit Player Data")
cols = ["Name","Value"] + [c for c in ("Position","FTPS","Bracket") if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True)

if "FTPS" not in edited.columns: edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]

players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

# --- Per-bracket sliders ---
brackets = sorted(edited["Bracket"].dropna().unique())
bracket_rand  = {}
bracket_usage = {}
if brackets:
    st.sidebar.markdown("### FTPS Randomness by Bracket")
    for b in brackets:
        bracket_rand[b] = st.sidebar.slider(
            f"Bracket {b} FTPS Random %", 0,100,0,5, key=f"rand_{b}"
        )
    st.sidebar.markdown("### Max Usage % by Bracket")
    for b in brackets:
        bracket_usage[b] = st.sidebar.slider(
            f"Bracket {b} Max Usage %", 0,100,100,5, key=f"use_{b}"
        )

# --- bracket constraint check ---
bracket_fail = False
if use_bracket_constraints and "Bracket" not in edited.columns:
    st.sidebar.warning("⚠️ No 'Bracket' column but Bracket Constraints is on.")
    bracket_fail = True

# --- Read target profile (Closest FTP Match) ---
target_values = None
if solver_mode=="Closest FTP Match" and template_file and format_name:
    prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
    raw  = prof.iloc[:,0].dropna().tolist()
    vals = [float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace(".","",1).isdigit()]
    if len(vals)<team_size:
        st.error(f"❌ Profile has fewer than {team_size} rows."); st.stop()
    target_values = vals[:team_size]

def add_bracket_constraints(prob,xvars):
    if use_bracket_constraints and not bracket_fail:
        groups={}
        for p in players:
            b=p.get("Bracket")
            if b: groups.setdefault(b,[]).append(xvars[p["Name"]])
        for grp in groups.values(): prob+=lpSum(grp)<=1

def add_usage_constraints(prob,xvars):
    for p in players:
        nm   = p["Name"]
        if nm in include_players: continue
        b    = p.get("Bracket")
        pct  = bracket_usage.get(b, global_max_usage)
        cap  = math.floor(num_teams * pct/100)
        used = sum(1 for prev in prev_sets if nm in prev)
        prob += (used + xvars[nm] <= cap, f"Usage_{nm}")

# --- Optimize Teams ---
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams=[]; prev_sets=[]

    # --- Budget Usage mode ---
    if solver_mode=="Maximize Budget Usage":
        upper=budget
        for _ in range(num_teams):
            prob=LpProblem("opt",LpMaximize)
            x={p["Name"]:LpVariable(p["Name"],cat="Binary") for p in players}
            cost=lpSum(x[n]*next(q["Value"] for q in players if q["Name"]==n) for n in x)
            prob+=cost
            prob+=lpSum(x.values())==team_size
            prob+=cost<=upper
            add_bracket_constraints(prob,x)
            add_usage_constraints(prob,x)
            for n in include_players: prob+=x[n]==1
            for n in exclude_players: prob+=x[n]==0
            for prev in prev_sets: prob+=lpSum(x[n] for n in prev)<=team_size-diff_count
            prob.solve()
            team=[p for p in players if x[p["Name"]].value()==1]
            all_teams.append(team); prev_sets.append({p["Name"] for p in team})
            upper=sum(p["Value"] for p in team)-0.001

    # --- Maximize FTPS mode ---
    elif solver_mode=="Maximize FTPS":
        for idx in range(num_teams):
            # Team 1: raw base_FTPS
            if idx==0:
                ftps_vals={p["Name"]:p["base_FTPS"] for p in players}
            else:
                ftps_vals={}
                for p in players:
                    b=p.get("Bracket")
                    r=bracket_rand.get(b,0)
                    ftps_vals[p["Name"]] = p["base_FTPS"]*(1+random.uniform(-r/100,r/100))

            prob=LpProblem("opt",LpMaximize)
            x={p["Name"]:LpVariable(p["Name"],cat="Binary") for p in players}
            prob+=lpSum(x[n]*ftps_vals[n] for n in x)
            prob+=lpSum(x.values())==team_size
            prob+=lpSum(x[n]*next(q["Value"] for q in players if q["Name"]==n) for n in x) <= budget
            add_bracket_constraints(prob,x)
            add_usage_constraints(prob,x)
            for n in include_players: prob+=x[n]==1
            for n in exclude_players: prob+=x[n]==0
            for prev in prev_sets: prob+=lpSum(x[n] for n in prev)<=team_size-diff_count
            prob.solve()
            team=[p for p in players if x[p["Name"]].value()==1]
            all_teams.append(team); prev_sets.append({p["Name"] for p in team})

    # --- Closest FTP Match mode ---
    else:
        for _ in range(num_teams):
            slots=[None]*team_size
            used_brackets=set(); used_names=set()
            # include players
            for n in include_players:
                p0=next(p for p in players if p["Name"]==n)
                diffs=[(i,abs(p0["Value"]-target_values[i])) for i in range(team_size) if slots[i] is None]
                best_i=min(diffs,key=lambda x:x[1])[0]
                slots[best_i]=p0; used_names.add(p0["Name"])
                if use_bracket_constraints and p0.get("Bracket"):
                    used_brackets.add(p0["Bracket"])
            def usage_ok(p):
                if p["Name"] in include_players: return True
                b=p.get("Bracket")
                pct=bracket_usage.get(b,global_max_usage)
                cap=math.floor(num_teams*pct/100)
                used=sum(1 for prev in prev_sets if p["Name"] in prev)
                return used<cap
            # greedy fill
            for i in range(team_size):
                if slots[i] is not None: continue
                tgt=target_values[i]
                cands=[p for p in players
                       if p["Name"] not in used_names
                       and p["Name"] not in exclude_players
                       and (not use_bracket_constraints or p.get("Bracket") not in used_brackets)
                       and usage_ok(p)]
                if not cands: break
                pick=min(cands,key=lambda p:abs(p["Value"]-tgt))
                slots[i]=pick; used_names.add(pick["Name"])
                if use_bracket_constraints and pick.get("Bracket"):
                    used_brackets.add(pick["Bracket"])
            cost=sum(p["Value"] for p in slots if p)
            if cost>budget:
                st.error(f"❌ Budget exceeded ({cost:.2f}>{budget:.2f})."); st.stop()
            names_set={p["Name"] for p in slots if p}
            if all(len(names_set&prev)<=team_size-diff_count for prev in prev_sets):
                all_teams.append(slots); prev_sets.append(names_set)
                if len(all_teams)==num_teams: break

    # --- Write & display ---
    if not all_teams:
        st.error("❌ Geen teams gecreëerd; controleer je instellingen."); st.stop()
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        for i,team in enumerate(all_teams,1):
            df_t=pd.DataFrame(team)
            df_t["Selectie (%)"]=df_t["Name"].apply(
                lambda n: round(
                    sum(1 for t in all_teams if any(p["Name"]==n for p in t))
                    /len(all_teams)*100,1))
            df_t.to_excel(writer,sheet_name=f"Team{i}",index=False)
    buf.seek(0)
    for i,team in enumerate(all_teams,1):
        with st.expander(f"Team {i}"):
            df_t=pd.DataFrame(team)
            df_t["Selectie (%)"]=df_t["Name"].apply(
                lambda n: round(
                    sum(1 for t in all_teams if any(p["Name"]==n for p in t))
                    /len(all_teams)*100,1))
            st.dataframe(df_t)
    st.download_button("📥 Download All Teams (Excel)",buf,
                       file_name="all_teams.xlsx",
                       mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet")
