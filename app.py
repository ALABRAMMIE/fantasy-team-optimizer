import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random, re, math
from io import BytesIO

st.title("Fantasy Team Optimizer")

# --- Sidebar Setup ---
sport_options = ["-- Choose --","Cycling","Speed Skating","Formula 1","Stock Exchange",
                 "Tennis","MotoGP","Football","Darts","Cyclocross","Golf","Snooker",
                 "Olympics","Basketball","Dakar Rally","Skiing","Rugby","Biathlon",
                 "Handball","Cross Country","Baseball","Ice Hockey",
                 "American Football","Ski Jumping","MMA","Entertainment","Athletics"]
sport = st.sidebar.selectbox("Sport", sport_options)

# Reset state when sport changes
if "selected_sport" not in st.session_state or st.session_state.selected_sport != sport:
    st.session_state.clear()
    st.session_state.selected_sport = sport

# Template upload
st.sidebar.markdown("### Profile Template (multi-sheet)")
template_file = st.sidebar.file_uploader("Upload Template", type="xlsx")

available_formats, format_name = [], None
if template_file:
    try:
        xl = pd.ExcelFile(template_file)
        available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
        if available_formats:
            format_name = st.sidebar.selectbox("Format", available_formats)
    except:
        st.sidebar.warning("⚠️ Can't read template sheets")

use_brackets = st.sidebar.checkbox("Use Bracket Constraints")

# --- Team Size Input (free, dynamic default) ---
default_team_size = 13
# If a template format is selected, infer team size from profile length
if template_file and format_name:
    try:
        prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
        vals = prof.iloc[:,0].dropna().tolist()
        default_team_size = len(vals)
    except:
        pass
team_size = st.sidebar.number_input(
    "Team Size",
    min_value=1,
    value=default_team_size,
    step=1,
)

solver = st.sidebar.radio("Solver", ["Maximize FTPS","Maximize Budget Usage","Closest FTP Match"])
num_teams = st.sidebar.number_input("Number of Teams", 1, 20, 1)
max_pct = st.sidebar.number_input("Max Player Frequency %", 0, 100, 100, 5)
max_occ = math.floor(max_pct/100 * num_teams)

# --- Player Upload & Edit ---
uploaded = st.file_uploader("Upload Players .xlsx", type="xlsx")
if not uploaded:
    st.info("Upload players file to continue.")
    st.stop()

df = pd.read_excel(uploaded)
if not {"Name","Value"}.issubset(df.columns):
    st.error("File must include columns: Name, Value")
    st.stop()

st.subheader("Edit Player Data")
cols = ["Name","Value"]
if "Position" in df.columns: cols.append("Position")
if "Rank FTPS" in df.columns: cols.append("Rank FTPS")
if "Bracket" in df.columns: cols.append("Bracket")
edited = st.data_editor(df[cols], use_container_width=True)

# Compute FTPS
if "Rank FTPS" in edited.columns:
    rank_map = {r: max(0,150-(r-1)*5) for r in range(1,31)}
    edited["FTPS"] = edited["Rank FTPS"].apply(lambda x: rank_map.get(int(x),0) if pd.notnull(x) else 0)
else:
    edited["FTPS"] = 0

# Bracket check
br_fail = False
if use_brackets and "Bracket" not in edited.columns:
    st.warning("⚠️ No Bracket column present; disabling bracket constraint")
    br_fail = True

players = edited.to_dict("records")

# --- Dynamic Max Budget Input ---
sorted_vals = sorted(edited["Value"], reverse=True)
budget_default = float(sum(sorted_vals[:team_size]))
budget = st.sidebar.number_input(
    "Max Budget",
    min_value=0.0,
    value=budget_default,
    step=0.1,
)

# Include/Exclude toggles
if "toggles" not in st.session_state: st.session_state.toggles = {}
incs = [n for n,v in st.session_state.toggles.items() if v=="✔"]
excs = [n for n,v in st.session_state.toggles.items() if v=="✖"]
include = st.sidebar.multiselect("Include", edited["Name"], default=nincs)
exclude = st.sidebar.multiselect("Exclude", edited["Name"], default=excs)

# Prepare profile values for Closest FTP Match
target = None
if solver=="Closest FTP Match" and template_file and format_name:
    prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
    raw = prof.iloc[:,0].dropna().tolist()
    vals = [float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace(".","",1).isdigit()]
    if len(vals) < team_size:
        st.error(f"Profile has {len(vals)} values; needs {team_size}")
    else:
        target = vals[:team_size]

# --- Optimize Button ---
optimize_clicked = st.sidebar.button("🚀 Optimize Teams")

# --- Optimize & Generate Teams ---
if optimize_clicked:
    freq = {p["Name"]:0 for p in players}
    all_teams = []

    def add_bracket_cons(prob, x):
        if use_brackets and not br_fail:
            buckets = {}
            for p in players:
                b = p.get("Bracket")
                if b: buckets.setdefault(b,[]).append(x[p["Name"]])
            for lst in buckets.values(): prob += lpSum(lst) <= 1

    # LP-based solvers
    if solver in ["Maximize FTPS","Maximize Budget Usage"]:
        prev = []
        for _ in range(num_teams):
            prob = LpProblem("opt", LpMaximize)
            x = {p["Name"]:LpVariable(p["Name"],cat="Binary") for p in players}
            # objective
            if solver=="Maximize FTPS": prob += lpSum(x[n]*next(p["FTPS"] for p in players if p["Name"]==n) for n in x)
            else: prob += lpSum(x[n]*next(p["Value"] for p in players if p["Name"]==n) for n in x)
            # constraints
            prob += lpSum(x[n] for n in x)==team_size
            prob += lpSum(x[n]*next(p["Value"] for p in players if p["Name"]==n) for n in x)<=budget
            add_bracket_cons(prob,x)
            for n in include: prob += x[n]==1
            for n in exclude: prob += x[n]==0
            for n,c in freq.items():
                if n not in include and n not in exclude and c>=max_occ:
                    prob += x[n]==0
            for t in prev: prob += lpSum(x[n] for n in t)<=team_size-1

            prob.solve()
            status = prob.status
            sel_names = [n for n in x if x[n].value()==1]

            # fallback: feasibility LP if under-selected
            if status!=1 or len(sel_names)<team_size:
                fea = LpProblem("fea",LpMaximize)
                x2 = {n:LpVariable(n,cat="Binary") for n in x}
                fea += lpSum(x2[n] for n in x2)
                fea += lpSum(x2[n] for n in x2)==team_size
                fea += lpSum(x2[n]*next(p["Value"] for p in players if p["Name"]==n) for n in x2)<=budget
                add_bracket_cons(fea,x2)
                for n in include: fea += x2[n]==1
                for n in exclude: fea += x2[n]==0
                fea.solve()
                sf = [n for n in x2 if x2[n].value()==1]
                if fea.status==1 and len(sf)==team_size:
                    team = [p for p in players if p["Name"] in sf]
                else:
                    # greedy fallback
                    team, used = [], set()
                    for n in include:
                        item = next(p for p in players if p["Name"]==n)
                        team.append(item); used.add(n)
                    keyf = (lambda p:-p["Value"]) if solver=="Maximize Budget Usage" else (lambda p:-p.get("FTPS",0))
                    while len(team)<team_size:
                        cands = [p for p in players if p["Name"] not in used]
                        if use_brackets:
                            used_b = {q.get("Bracket") for q in team if q.get("Bracket")}
                            cands = [p for p in cands if p.get("Bracket") not in used_b]
                        cands.sort(key=keyf)
                        pick = cands[0]
                        team.append(pick); used.add(pick["Name"])
            else:
                team = [p for p in players if p["Name"] in sel_names]

            prev.append([p["Name"] for p in team])
            for p in team: freq[p["Name"]]+=1
            all_teams.append(team)

    # Greedy solver
    else:
        seen={}
        for _ in range(num_teams*100):
            random.shuffle(players)
            avail=[p for p in players if p["Name"] not in exclude]
            sel,used,brs=[],set(),set()
            for n in include:
                itm=next((p for p in avail if p["Name"]==n),None)
                if itm: sel.append(itm); used.add(n)
            if use_brackets and not br_fail:
                for p in avail:
                    b=p.get("Bracket")
                    if b and p["Name"] not in used and b not in brs:
                        sel.append(p); used.add(p["Name"]); brs.add(b); break
            for tgt in (target or [0]*team_size)[len(sel):]:
                cands=[p for p in avail if p["Name"] not in used and (p["Name"] in include or p["Name"] in exclude or freq[p["Name"]]<max_occ)]
                if use_brackets: cands=[p for p in cands if p.get("Bracket") not in brs]
                if not cands: cands=[p for p in avail if p["Name"] not in used]
                cands.sort(key=lambda x: abs(x["Value"]-tgt))
                if not cands: break
                pick=cands[0]
                sel.append(pick); used.add(pick["Name"]); brs.add(pick.get("Bracket"))
            if len(sel)==team_size:
                err=sum((sel[i]["Value"]-(target or [0])[i])**2 for i in range(team_size))
                key=tuple(p["Name"] for p in sel)
                if key not in seen or err<seen[key][0]: seen[key]=(err,sel)
            if len(seen)>=num_teams: break
        for _,team in sorted(seen.values(), key=lambda x:x[0])[:num_teams]:
            for p in team: freq[p["Name"]]+=1
            all_teams.append(team)

    # Download All Teams
    buf=BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for i,team in enumerate(all_teams,1): pd.DataFrame(team).to_excel(w,sheet_name=f"Team{i}",index=False)
    buf.seek(0)
    for i,team in enumerate(all_teams,1):
        with st.expander(f"Team {i}"): st.dataframe(pd.DataFrame(team))
    st.download_button("📥 Download All Teams (Excel)", buf,
                       file_name="all_teams.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
