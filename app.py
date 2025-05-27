
import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum
import random
import re

st.title("Fantasy Team Optimizer")

sport_options = [
    "-- Choose a sport --",
    "Cycling", "Speed Skating", "Formula 1", "Stock Exchange", "Tennis",
    "MotoGP", "Football", "Darts", "Cyclocross", "Golf", "Snooker",
    "Olympics", "Basketball", "Dakar Rally", "Skiing", "Rugby", "Biathlon",
    "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment", "Athletics"
]

# Sport selector
sport = st.sidebar.selectbox("Select a sport", sport_options)

# Reset state on sport change
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for key in list(st.session_state.keys()):
        if key != "selected_sport":
            del st.session_state[key]
    st.session_state.selected_sport = sport

# Upload profile template
st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template (multi-sheet)",
    type=["xlsx"], key="template_upload_key"
)

# Detect formats
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

# Bracket constraint toggle
use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")

# Budget and team size
budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)

# Solver selection
solver_mode = st.sidebar.radio("Solver Objective", [
    "Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"
])

# Player upload
uploaded_file = st.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    # Validate columns
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("❌ File must include 'Name' and 'Value'.")
    else:
        st.subheader("📋 Edit Player Data")
        editable = ["Name", "Value"]
        if "Position" in df.columns:
            editable.append("Position")
        if "Rank FTPS" in df.columns:
            editable.append("Rank FTPS")
        if "Bracket" in df.columns:
            editable.append("Bracket")
        edited_df = st.data_editor(df[editable], use_container_width=True)

        # Compute FTPS
        if "Rank FTPS" in edited_df.columns:
            rk = {r: max(0, 150 - (r - 1) * 5) for r in range(1, 31)}
            edited_df["FTPS"] = edited_df["Rank FTPS"].apply(
                lambda x: rk.get(int(x), 0) if pd.notnull(x) else 0
            )
        else:
            edited_df["FTPS"] = 0

        # Bracket presence check
        bracket_fail = False
        if use_bracket_constraints and "Bracket" not in edited_df.columns:
            st.warning("⚠️ Bracket enabled but no 'Bracket' column found.")
            bracket_fail = True

        players = edited_df.to_dict("records")

        if "toggle_choices" not in st.session_state:
            st.session_state.toggle_choices = {}

        # Sidebar include/exclude
        default_inc = [n for n,v in st.session_state.toggle_choices.items() if v=="✔"]
        default_exc = [n for n,v in st.session_state.toggle_choices.items() if v=="✖"]
        include_players = st.sidebar.multiselect("Players to INCLUDE", edited_df["Name"], default=default_inc)
        exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited_df["Name"], default=default_exc)

        optimize_clicked = st.sidebar.button("🚀 Optimize Team")

        # Prepare target for Closest FTP Match
        target_values = None
        if solver_mode=="Closest FTP Match" and template_file and format_name:
            try:
                prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
                raw = prof.iloc[:,0].dropna().tolist()
                vals = [float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace(".","",1).isdigit()]
                if len(vals) < team_size:
                    st.error(f"❌ Profile has fewer than {team_size} rows.")
                else:
                    target_values = vals[:team_size]
            except Exception as e:
                st.error(f"❌ Failed to read profile: {e}")

        # Optimization
        if optimize_clicked:
            result_df = None

            # Helper: add bracket constraints to LP
            def add_bracket_constraints(prob, x_vars):
                if use_bracket_constraints and not bracket_fail:
                    brackets = {}
                    for p in players:
                        b = p.get("Bracket")
                        if b:
                            brackets.setdefault(b, []).append(x_vars[p["Name"]])
                    for vars_list in brackets.values():
                        prob += lpSum(vars_list) <= 1

            # Closest FTP Match
            if solver_mode=="Closest FTP Match":
                if not target_values:
                    st.warning("⚠️ No target values loaded.")
                elif use_bracket_constraints and bracket_fail:
                    st.warning("⚠️ Bracket enabled but missing column.")
                else:
                    avail = [p for p in players if p["Name"] not in exclude_players]
                    sel, used = [], set()
                    # force includes
                    for n in include_players:
                        for p in avail:
                            if p["Name"]==n:
                                sel.append(p); used.add(n); break
                    # bracket picks
                    if use_bracket_constraints:
                        used_br = set()
                        for p in avail:
                            br = p.get("Bracket")
                            if br and p["Name"] not in used and br not in used_br:
                                sel.append(p); used.add(p["Name"]); used_br.add(br)
                                if len(sel)==team_size: break
                        for tgt in target_values[len(sel):]:
                            cands = sorted(
                                [p for p in avail if p["Name"] not in used and p.get("Bracket") not in used_br],
                                key=lambda x: abs(x["Value"]-tgt)
                            )
                            if cands:
                                c=cands[0]; sel.append(c); used.add(c["Name"]); used_br.add(c.get("Bracket"))
                    # greedy fill
                    for tgt in target_values[len(sel):]:
                        cands = sorted([p for p in avail if p["Name"] not in used],
                                       key=lambda x: abs(x["Value"]-tgt))
                        if cands:
                            c=cands[0]; sel.append(c); used.add(c["Name"])
                    if len(sel)==team_size:
                        result_df=pd.DataFrame(sel); st.session_state["result_df"]=result_df
                    else:
                        st.error(f"❌ Only {len(sel)} players selected.")
            # Maximize FTPS
            elif solver_mode=="Maximize FTPS":
                prob = LpProblem("MaxFTPS", LpMaximize)
                x = {p["Name"]: LpVariable(p["Name"],cat="Binary") for p in players}
                prob += lpSum(x[p["Name"]]*p["FTPS"] for p in players)
                prob += lpSum(x[p["Name"]] for p in players)==team_size
                prob += lpSum(x[p["Name"]]*p["Value"] for p in players)<=budget
                add_bracket_constraints(prob,x)
                for n in include_players: prob += x[n]==1
                for n in exclude_players: prob += x[n]==0
                prob.solve()
                sel=[p for p in players if x[p["Name"]].value()==1]
                result_df=pd.DataFrame(sel); st.session_state["result_df"]=result_df
            # Maximize Budget Usage
            elif solver_mode=="Maximize Budget Usage":
                prob = LpProblem("MaxBudg", LpMaximize)
                x = {p["Name"]: LpVariable(p["Name"],cat="Binary") for p in players}
                prob += lpSum(x[p["Name"]]*p["Value"] for p in players)
                prob += lpSum(x[p["Name"]] for p in players)==team_size
                prob += lpSum(x[p["Name"]]*p["Value"] for p in players)<=budget
                add_bracket_constraints(prob,x)
                for n in include_players: prob += x[n]==1
                for n in exclude_players: prob += x[n]==0
                prob.solve()
                sel=[p for p in players if x[p["Name"]].value()==1]
                result_df=pd.DataFrame(sel); st.session_state["result_df"]=result_df

        # Display result
        if "result_df" in st.session_state:
            df_res=st.session_state["result_df"]
            st.subheader("🎯 Optimized Team")
            togg=[]
            for _,r in df_res.iterrows():
                default=st.session_state.toggle_choices.get(r["Name"],"–")
                choice=st.radio(r["Name"],["✔","✖","–"],horizontal=True,key=f"tog_{r['Name']}",index=["✔","✖","–"].index(default))
                st.session_state.toggle_choices[r["Name"]]=choice
                togg.append(choice)
            if "🔧" in df_res.columns: df_res=df_res.drop(columns=["🔧"])
            df_res.insert(0,"🔧",togg)
            st.dataframe(df_res)
