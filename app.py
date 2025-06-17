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

# --- Tour Substitutes Option (Cycling Only) ---
tour_mode = False
if sport == "Cycling":
    tour_mode = st.sidebar.checkbox("Enable Tour Event Substitutes")
    if tour_mode:
        tour_budget = st.sidebar.number_input("Tour Substitute Budget", value=25.0)
        tour_team_size = st.sidebar.number_input("Tour Substitute Team Size", min_value=1, value=3)

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
    if m: default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", min_value=1, value=default_team_size)
solver_mode = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams  = st.sidebar.number_input("Number of Teams", min_value=1, max_value=100, value=1)
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
    st.error("❌ File must include 'Name' and 'Value'.")
    st.stop()

st.subheader("📋 Edit Player Data")
cols = ["Name","Value"] + [c for c in ("Position","FTPS","Bracket") if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True, num_rows='dynamic')
if "FTPS" not in edited.columns: edited["FTPS"] = 0
edited["base_FTPS"] = edited["FTPS"]
players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

# --- Bracket sliders ---
brackets = sorted(edited["Bracket"].dropna().unique())
if use_bracket_constraints and not brackets:
    st.sidebar.warning("⚠️ Bracket Constraints on but no 'Bracket' column found.")
bracket_min_count, bracket_max_count = {}, {}
if brackets:
    with st.sidebar.expander("Usage Count by Bracket", expanded=False):
        for b in brackets:
            bracket_min_count[b] = st.number_input(f"Bracket {b} Min per team",0,team_size,0)
            bracket_max_count[b] = st.number_input(f"Bracket {b} Max per team",0,team_size,team_size)

# --- Closest FTP target ---
target_values=None
if solver_mode=="Closest FTP Match" and template_file and format_name:
    prof=pd.read_excel(template_file,sheet_name=format_name,header=None)
    raw=prof.iloc[:,0].dropna().tolist()
    vals=[float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace('.', '',1).isdigit()]
    if len(vals)<team_size: st.error(f"❌ Profile has fewer than {team_size} rows."); st.stop()
    target_values=vals[:team_size]

# --- Helpers ---
def add_bracket_constraints(prob,x):
    if use_bracket_constraints:
        for b in brackets: prob+=lpSum(x[p['Name']] for p in players if p.get('Bracket')==b)<=1, f"UniqueBracket_{b}"
def add_composition_constraints(prob,x):
    for b in brackets:
        mn, mx = bracket_min_count.get(b,0), bracket_max_count.get(b,team_size)
        lst=[x[p['Name']] for p in players if p.get('Bracket')==b]
        if mn>0: prob+=lpSum(lst)>=mn, f"MinBracket_{b}"
        if mx<team_size: prob+=lpSum(lst)<=mx, f"MaxBracket_{b}"
def add_global_usage_cap(prob,x):
    cap=math.floor(num_teams*global_usage_pct/100)
    for p in players:
        if p['Name'] not in include_players: used=sum(1 for s in prev_sets if p['Name'] in s); prob+=used+x[p['Name']]<=cap, f"GlobalUse_{p['Name']}"
def add_min_diff(prob,x):
    for i,s in enumerate(prev_sets): prob+=lpSum(x[n] for n in s)<=team_size-diff_count, f"MinDiff_{i}"

# --- Optimize ---
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams, prev_sets, subs = [], [], []
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
        if sport=="Cycling" and tour_mode:
            rem=[p for p in players if p['Name'] not in {nm for t in all_teams for nm in [pp['Name'] for pp in t]}]
            sub=LpProblem("subs",LpMaximize)
            xs={p['Name']:LpVariable(p['Name'],cat='Binary') for p in rem}
            sub+=lpSum(xs[n]*next(q['Value'] for q in rem if q['Name']==n) for n in xs)
            sub+=lpSum(xs.values())==tour_team_size
            sub+=lpSum(xs[n]*next(q['Value'] for q in rem if q['Name']==n) for n in xs)<=tour_budget
            sub.solve(); subs=[p for p in rem if xs[p['Name']].value()==1]
    elif solver_mode=="Maximize FTPS":
        for idx in range(num_teams):
            ftps_vals={p['Name']:p['base_FTPS'] for p in players} if idx==0 else {p['Name']:p['base_FTPS']*(1+random.uniform(-ftps_rand_pct/100,ftps_rand_pct/100)) for p in players}
            prob=LpProblem(f"opt_f_{idx}",LpMaximize)
            x={p['Name']:LpVariable(p['Name'],cat='Binary') for p in players}
            prob+=lpSum(x[n]*ftps_vals[n] for n in x)
            prob+=lpSum(x.values())==team_size
            prob+=lpSum(x[n]*next(q['Value'] for q in players if q['Name']==n) for n in x)<=budget
            add_bracket_constraints(prob,x); add_composition_constraints(prob,x)
            add_global_usage_cap(prob,x); add_min_diff(prob,x)
            for n in include_players: prob+=x[n]==1
            for n in exclude_players: prob+=x[n]==0
            prob.solve(); team=[{**p,'Adjusted FTPS':ftps_vals[p['Name']]} for p in players if x[p['Name']].value()==1]
            all_teams.append(team); prev_sets.append({p['Name'] for p in team})
    else:
        cap=math.floor(num_teams*global_usage_pct/100)
        for _ in range(num_teams):
            slots,ub,un=[None]*team_size,set(),set()
            for n in include_players:
                p0=next(p for p in players if p['Name']==n)
                diffs=[(i,abs(p0['Value']-target_values[i])) for i in range(team_size) if slots[i] is None]
                bi=min(diffs,key=lambda x:x[1])[0];slots[bi]=p0;un.add(n);ub.add(p0['Bracket']) if use_bracket_constraints and p0.get('Bracket') else None
            for i in range(team_size):
                if slots[i]:continue
                tgt=target_values[i]
                cands=[p for p in players if p['Name'] not in un and p['Name'] not in exclude_players and (not use_bracket_constraints or p.get('Bracket') not in ub) and sum(1 for s in prev_sets if p['Name'] in s)<cap]
                pick=min(cands,key=lambda p:abs(p['Value']-tgt));slots[i]=pick;un.add(pick['Name']);ub.add(pick['Bracket']) if use_bracket_constraints and pick.get('Bracket') else None
            team=[p for p in slots if p];all_teams.append(team);prev_sets.append({p['Name'] for p in team})
    # --- Display ---
    for i,team in enumerate(all_teams,1):
        with st.expander(f"Team {i}"):
            df_main=pd.DataFrame(team);df_main['Role']='Main'
            if i==1 and subs:
                df_sub=pd.DataFrame(subs);df_sub['Role']='Sub'
                df_t=pd.concat([df_main,df_sub],ignore_index=True)
            else:df_t=df_main
            df_t['Selectie (%)']=df_t['Name'].apply(lambda n:round(sum(1 for t in all_teams if any(p['Name']==n for p in t))/len(all_teams)*100,1))
            st.dataframe(df_t.style.apply(lambda r:['background-color: lightyellow' if r['Role']=='Sub' else '' for _ in r],axis=1))
    # --- Download ---
    merged=[]
    for idx,team in enumerate(all_teams,1):
        df_t=pd.DataFrame(team);df_t['Team']=idx;merged.append(df_t)
    merged_df=pd.concat(merged,ignore_index=True)
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:merged_df.to_excel(w,index=False,sheet_name='All Teams')
    buf.seek(0)
    st.download_button('📥 Download All Teams',buf,'all_teams.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
