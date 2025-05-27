
import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
import re
from io import BytesIO

st.title("Fantasy Team Optimizer")

sport_options = [
    "-- Choose a sport --",
    "Cycling", "Speed Skating", "Formula 1", "Stock Exchange", "Tennis",
    "MotoGP", "Football", "Darts", "Cyclocross", "Golf", "Snooker",
    "Olympics", "Basketball", "Dakar Rally", "Skiing", "Rugby", "Biathlon",
    "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment", "Athletics"
]

sport = st.sidebar.selectbox("Select a sport", sport_options)

if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for key in list(st.session_state.keys()):
        if key != "selected_sport":
            del st.session_state[key]
    st.session_state.selected_sport = sport

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
    except:
        st.sidebar.warning("⚠️ Unable to read sheets from template.")

use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")

budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)

solver_mode = st.sidebar.radio("Solver Objective", [
    "Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"
])

num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=10, value=1, step=1)

uploaded_file = st.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
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

        if "Rank FTPS" in edited_df.columns:
            rk = {r: max(0, 150 - (r - 1) * 5) for r in range(1, 31)}
            edited_df["FTPS"] = edited_df["Rank FTPS"].apply(
                lambda x: rk.get(int(x), 0) if pd.notnull(x) else 0
            )
        else:
            edited_df["FTPS"] = 0

        bracket_fail = False
        if use_bracket_constraints and "Bracket" not in edited_df.columns:
            st.warning("⚠️ Bracket enabled but no 'Bracket' column found.")
            bracket_fail = True

        players = edited_df.to_dict("records")

        if "toggle_choices" not in st.session_state:
            st.session_state.toggle_choices = {}

        default_inc = [n for n,v in st.session_state.toggle_choices.items() if v=="✔"]
        default_exc = [n for n,v in st.session_state.toggle_choices.items() if v=="✖"]
        include_players = st.sidebar.multiselect("Players to INCLUDE", edited_df["Name"], default=default_inc)
        exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited_df["Name"], default=default_exc)

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

        if st.sidebar.button("🚀 Optimize Teams"):
            all_teams = []

            def add_bracket_constraints(prob, x_vars):
                if use_bracket_constraints and not bracket_fail:
                    brackets = {}
                    for p in players:
                        b = p.get("Bracket")
                        if b:
                            brackets.setdefault(b, []).append(x_vars[p["Name"]])
                    for vars_list in brackets.values():
                        prob += lpSum(vars_list) <= 1

            if solver_mode in ["Maximize FTPS", "Maximize Budget Usage"]:
                prev_teams = []
                for _ in range(num_teams):
                    prob = (LpProblem("MaxFTPS", LpMaximize) if solver_mode=="Maximize FTPS"
                            else LpProblem("MaxBudget", LpMaximize))
                    x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
                    if solver_mode=="Maximize FTPS":
                        prob += lpSum(x[p["Name"]]*p["FTPS"] for p in players)
                    else:
                        prob += lpSum(x[p["Name"]]*p["Value"] for p in players)
                    prob += lpSum(x[p["Name"]] for p in players)==team_size
                    prob += lpSum(x[p["Name"]]*p["Value"] for p in players)<=budget
                    add_bracket_constraints(prob, x)
                    for n in include_players: prob += x[n]==1
                    for n in exclude_players: prob += x[n]==0
                    for t in prev_teams: prob += lpSum(x[name] for name in t)<=team_size-1
                    prob.solve()
                    sel=[p for p in players if x[p["Name"]].value()==1]
                    prev_teams.append([p["Name"] for p in sel])
                    all_teams.append(sel)

            elif solver_mode=="Closest FTP Match":
                seen={}
                attempts=num_teams*100
                for _ in range(attempts):
                    random.shuffle(players)
                    avail=[p for p in players if p["Name"] not in exclude_players]
                    sel,used=[],set()
                    for n in include_players:
                        for p in avail:
                            if p["Name"]==n:
                                sel.append(p); used.add(n); break
                    if use_bracket_constraints:
                        br_used=set()
                        for p in avail:
                            b=p.get("Bracket")
                            if b and p["Name"] not in used and b not in br_used:
                                sel.append(p); used.add(p["Name"]); br_used.add(b)
                                if len(sel)==team_size: break
                    for tgt in target_values[len(sel):]:
                        cands=sorted([p for p in avail if p["Name"] not in used], key=lambda x:abs(x["Value"]-tgt))
                        if cands:
                            c=cands[0]; sel.append(c); used.add(c["Name"])
                    if len(sel)==team_size:
                        err=sum((sel[i]["Value"]-target_values[i])**2 for i in range(team_size))
                        key=tuple(p["Name"] for p in sel)
                        if key not in seen or err<seen[key][0]:
                            seen[key]=(err,sel)
                    if len(seen)>=num_teams: break
                best=sorted(seen.values(),key=lambda x:x[0])[:num_teams]
                all_teams=[team for err,team in best]

            # Create Excel in-memory with openpyxl
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for idx, team in enumerate(all_teams):
                    pd.DataFrame(team).to_excel(writer, sheet_name=f"Team{idx+1}", index=False)
            output.seek(0)

            for idx, team in enumerate(all_teams):
                df_t=pd.DataFrame(team)
                with st.expander(f"Team {idx+1}"):
                    st.dataframe(df_t)

            st.download_button(
                "📥 Download All Teams (Excel)",
                output,
                file_name="all_teams.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
