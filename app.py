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
    xl = pd.ExcelFile(template_file)
    available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
    if available_formats:
        format_name = st.sidebar.selectbox("Select Format", available_formats)

# --- Core constraints ---
use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")
budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m: default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", min_value=1, value=default_team_size)
solver_mode = st.sidebar.radio(
    "Solver Objective", ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=100, value=1)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)", min_value=0, max_value=team_size, value=1
)

# --- FTPS randomness ---
ftps_rand_pct = st.sidebar.slider(
    "FTPS Randomness % for subsequent teams", 0, 100, 0, 5,
    help="± this percent noise on FTPS for teams 2…N"
)

# --- Global usage cap ---
global_usage_pct = st.sidebar.slider(
    "Global Max Usage % per player (across all teams)", 0, 100, 100, 5,
    help="Max fraction of teams any player can appear in (INCLUDE still forces 100%)."
)

# --- Upload & Edit Players ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

df = pd.read_excel(uploaded_file)
if not {"Name","Value"}.issubset(df.columns):
    st.error("File must include 'Name' and 'Value'.")
    st.stop()

st.subheader("📋 Edit Player Data")
cols = ["Name","Value"] + [c for c in ("Position","FTPS","Bracket") if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True, num_rows='dynamic')
if "FTPS" not in edited.columns: edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]

players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])
brackets = sorted(edited["Bracket"].dropna().unique())
if use_bracket_constraints and not brackets:
    st.sidebar.warning("⚠️ Bracket Constraints on but no 'Bracket' column found.")

# Per-bracket sliders
bracket_min_count, bracket_max_count = {}, {}
if brackets:
    with st.sidebar.expander("Usage Count by Bracket"):
        for b in brackets:
            bracket_min_count[b] = st.number_input(f"{b} min per team", 0, team_size, 0)
            bracket_max_count[b] = st.number_input(f"{b} max per team", 0, team_size, team_size)

# Constraint helpers
def add_bracket_constraints(prob,x):
    if use_bracket_constraints:
        for b in brackets: prob += lpSum(x[p['Name']] for p in players if p.get('Bracket')==b)<=1, f"Unique_{b}"
def add_composition_constraints(prob,x):
    for b in brackets:
        mn, mx = bracket_min_count.get(b,0), bracket_max_count.get(b,team_size)
        lst=[x[p['Name']] for p in players if p.get('Bracket')==b]
        if mn>0: prob+=lpSum(lst)>=mn, f"Min_{b}"
        if mx<team_size: prob+=lpSum(lst)<=mx, f"Max_{b}"
def add_global_usage_cap(prob,x):
    cap=math.floor(num_teams*global_usage_pct/100)
    for p in players:
        if p['Name'] not in include_players: prob+=sum(1 for s in prev_sets if p['Name'] in s)+x[p['Name']]<=cap, f"Use_{p['Name']}"
def add_min_diff(prob,x):
    for i,s in enumerate(prev_sets): prob+=lpSum(x[n] for n in s)<=team_size-diff_count, f"Diff_{i}"

# Optimize
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams, prev_sets, subs = [],[],[]
    # Budget
    if solver_mode=="Maximize Budget Usage":
        upper=budget
        for _ in range(num_teams):
            prob=LpProblem("opt_b",LpMaximize)
            x={p['Name']:LpVariable(p['Name'],cat='Binary') for p in players}
            prob+=lpSum(x[n]*next(q['Value'] for q in players if q['Name']==n) for n in x)
            prob+=lpSum(x.values())==team_size
            prob+=lpSum(x[n]*next(q['Value'] for q in players if q['Name']==n) for n in x)<=upper
            add_bracket_constraints(prob,x); add_composition_constraints(prob,x)
            add_global_usage_cap(prob,x); add_min_diff(prob,x)
            for n in include_players: prob+=x[n]==1
            for n in exclude_players: prob+=x[n]==0
            prob.solve()
            team=[p for p in players if x[p['Name']].value()==1]
            all_teams.append(team); prev_sets.append({p['Name'] for p in team}); upper=sum(p['Value'] for p in team)-1e-3
        # Subs under T1
        rem=[p for p in players if p['Name'] not in {n for t in all_teams for n in [pp['Name'] for pp in t]}]
        prob=LpProblem("subs",LpMaximize)
        xs={p['Name']:LpVariable(p['Name'],cat='Binary') for p in rem}
        prob+=lpSum(xs[n]*next(q['Value'] for q in rem if q['Name']==n) for n in xs)
        prob+=lpSum(xs.values())==min(ftps_rand_pct, len(rem))
        prob+=lpSum(xs[n]*next(q['Value'] for q in rem if q['Name']==n) for n in xs)<=budget*0.2
        prob.solve(); subs=[p for p in rem if xs[p['Name']].value()==1]
    # FTPS
    elif solver_mode=="Maximize FTPS":
        for idx in range(num_teams):
            vals={p['Name']:p['base_FTPS'] for p in players} if idx==0 else {p['Name']:p['base_FTPS']*(1+random.uniform(-ftps_rand_pct/100,ftps_rand_pct/100)) for p in players}
            prob=LpProblem(f"opt_f_{idx}",LpMaximize)
            x={p['Name']:LpVariable(p['Name'],cat='Binary') for p in players}
            prob+=lpSum(x[n]*vals[n] for n in x)
            prob+=lpSum(x.values())==team_size
            prob+=lpSum(x[n]*next(q['Value'] for q in players if q['Name']==n) for n in x)<=budget
            add_bracket_constraints(prob,x); add_composition_constraints(prob,x)
            add_global_usage_cap(prob,x); add_min_diff(prob,x)
            for n in include_players: prob+=x[n]==1
            for n in exclude_players: prob+=x[n]==0
            prob.solve()
            team=[{**p,'Adjusted FTPS':vals[p['Name']]} for p in players if x[p['Name']].value()==1]
            all_teams.append(team); prev_sets.append({p['Name'] for p in team})
    # Closest match omitted
    # Display
    for i,team in enumerate(all_teams,1):
        with st.expander(f"Team {i}"):
            df_t=pd.DataFrame(team)
            dfs=[r for r in df_t.itertuples()]
            if i==1 and subs:
                df_sub=pd.DataFrame(subs)
                df_sub['Sub']=True; df_sub['Value']=''
                df_t=pd.concat([df_t,df_sub],ignore_index=True)
            df_t['Selectie (%)']=df_t['Name'].apply(lambda n:round(sum(n in [p['Name'] for p in t] for t in all_teams)/len(all_teams)*100,1))
            st.dataframe(df_t.style.apply(lambda r:['background-color: lightyellow' if 'Sub' in r and r['Sub'] else '' for _ in r],axis=1))
    # Download\ nbuf=BytesIO();pd.DataFrame([{'Team':r['Name']} for r in all_teams]);buf.seek(0)
    st.download_button('📥 Download All Teams',buf,'all_teams.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
