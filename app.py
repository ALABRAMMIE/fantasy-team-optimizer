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
        if k != "selected_sport": del st.session_state[k]
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
budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m: default_team_size = int(m.group(1))
team_size = st.sidebar.number_input(
    "Team Size", min_value=1, value=default_team_size, step=1
)
solver_mode = st.sidebar.radio(
    "Solver Objective", ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams = st.sidebar.number_input("Number of Teams", min_value=1, step=1, value=1)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)", min_value=0, max_value=team_size, value=1
)

# --- FTPS randomness ---
ftps_rand_pct = st.sidebar.slider(
    "FTPS Randomness % for subsequent teams", 0,100,0,5,
    help="±this percent noise on FTPS for teams 2…N"
)

# --- Global usage cap ---
global_usage_pct = st.sidebar.slider(
    "Global Max Usage % per player (across all teams)",
    0,100,100,5,
    help="Max fraction of teams any player can appear in (INCLUDE still forces 100%)."
)

# --- Tour Event Substitutes (Cycling only) ---
tour_mode = False
tour_budget = None
tour_team_size = None
if sport == "Cycling":
    tour_mode = st.sidebar.checkbox("Enable Tour Event Substitutes")
    if tour_mode:
        tour_budget = st.sidebar.number_input("Tour Substitute Budget", value=25.0)
        tour_team_size = st.sidebar.number_input(
            "Tour Substitute Team Size", min_value=1, value=3, step=1
        )

# --- Upload & Edit Players ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader(
    "Upload your Excel file (players)", type=["xlsx"]
)
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

try: df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Failed to read players file: {e}")
    st.stop()
if not {"Name","Value"}.issubset(df.columns):
    st.error("❌ File must include 'Name' and 'Value'.")
    st.stop()

st.subheader("📋 Edit Player Data")
cols = ["Name","Value"] + [c for c in ("Position","FTPS","Bracket") if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True)
if "FTPS" not in edited.columns: edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]
players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])
# collect brackets
brackets = sorted(edited["Bracket"].dropna().unique())
if use_bracket_constraints and not brackets:
    st.sidebar.warning("⚠️ Bracket Constraints on but no ‘Bracket’ column found.")
# per-bracket sliders
bracket_min_count, bracket_max_count = {},{}
if brackets:
    with st.sidebar.expander("Usage Count by Bracket",expanded=False):
        for b in brackets:
            bracket_min_count[b] = st.number_input(
                f"Bracket {b} Min picks per team",0,team_size,0,1,key=f"min_{b}"
            )
            bracket_max_count[b] = st.number_input(
                f"Bracket {b} Max picks per team",0,team_size,team_size,1,key=f"max_{b}"
            )
# read profile
target_values=None
if solver_mode=="Closest FTP Match" and template_file and format_name:
    try:
        prof=pd.read_excel(template_file,sheet_name=format_name,header=None)
        raw=prof.iloc[:,0].dropna().tolist()
        vals=[float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace(".","",1).isdigit()]
        if len(vals)<team_size:
            st.error(f"❌ Profile has fewer than {team_size} rows.")
            st.stop()
        target_values=vals[:team_size]
    except Exception as e:
        st.error(f"❌ Failed to read profile: {e}")
        st.stop()
# helpers
def add_bracket_constraints(prob,x):
    if use_bracket_constraints:
        for b in brackets:
            prob += lpSum(x[p["Name"]] for p in players if p.get("Bracket")==b)<=1, f"Unique_{b}"
def add_composition_constraints(prob,x):
    for b in brackets:
        mn=bracket_min_count.get(b,0); mx=bracket_max_count.get(b,team_size)
        members=[x[p["Name"]] for p in players if p.get("Bracket")==b]
        if mn>0: prob+=lpSum(members)>=mn,f"Min_{b}"
        if mx<team_size: prob+=lpSum(members)<=mx,f"Max_{b}"
def add_global_usage_cap(prob,x):
    if num_teams<=1: return
    cap=math.floor(num_teams*global_usage_pct/100)
    for p in players:
        nm=p["Name"]
        if nm in include_players: continue
        used=sum(1 for prev in prev_sets if nm in prev)
        prob+=(used+x[nm]<=cap,f"GlobalUse_{nm}")
def add_min_diff(prob,x):
    for idx,prev in enumerate(prev_sets): prob+=lpSum(x[n] for n in prev)<=team_size-diff_count,f"Diff_{idx}"
# optimize
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams=[]; prev_sets=[]
    if solver_mode=="Maximize Budget Usage":
        upper=budget
        for _ in range(num_teams):
            prob=LpProblem("opt_budget",LpMaximize)
            x={p["Name"]:LpVariable(p["Name"],cat="Binary") for p in players}
            prob+=lpSum(x[n]*next(q["Value"] for q in players if q["Name']==n) for n in x)
            prob+=lpSum(x.values())==team_size
            prob+=lpSum(x[n]*next(q["Value"] for q in players if q["Name']==n) for n in x)<=upper
            add_bracket_constraints(prob,x); add_composition_constraints(prob,x)
            add_global_usage_cap(prob,x); add_min_diff(prob,x)
            for n in include_players: prob+=x[n]==1
            for n in exclude_players: prob+=x[n]==0
            prob.solve(); assert prob.status==1, st.error("🚫 Infeasible under those constraints."), st.stop()
            team=[p for p in players if x[p["Name"]].value()==1]
            all_teams.append(team); prev_sets.append({p["Name"] for p in team})
            upper=sum(p["Value"] for p in team)-0.001
        # Tour substitutes
        if tour_mode and sport=="Cycling":
            # pick only from unselected players
            rem=[p for p in players if all(p["Name"] not in used for used in prev_sets)]
            prob=LpProblem("tour_subs",LpMaximize)
            xs={p["Name"]:LpVariable(p["Name"],cat="Binary") for p in rem}
            prob+=lpSum(xs[n]*next(q["Value"] for q in rem if q["Name']==n) for n in xs)
            prob+=lpSum(xs.values())==tour_team_size
            prob+=lpSum(xs[n]*next(q["Value"] for q in rem if q["Name']==n) for n in xs)<=tour_budget
            prob.solve(); subs=[p for p in rem if xs[p["Name"]].value()==1]
            st.subheader("🔄 Tour Substitutes")
            st.dataframe(pd.DataFrame(subs))
    # ... other solver branches unchanged ...
    # display and download as before
