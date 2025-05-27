import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
import re
from io import BytesIO
import math

st.title("Fantasy Team Optimizer")

sport_options = [
    "-- Choose a sport --", "Cycling", "Speed Skating", "Formula 1", "Stock Exchange",
    "Tennis", "MotoGP", "Football", "Darts", "Cyclocross", "Golf", "Snooker",
    "Olympics", "Basketball", "Dakar Rally", "Skiing", "Rugby", "Biathlon",
    "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment", "Athletics"
]

sport = st.sidebar.selectbox("Select a sport", sport_options)

# Reset state on sport change
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for k in list(st.session_state.keys()):
        if k != "selected_sport":
            del st.session_state[k]
    st.session_state.selected_sport = sport

# Upload profile template
st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template (multi-sheet)", type=["xlsx"], key="template_upload_key"
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

# Options
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
num_teams = st.sidebar.number_input("Number of Teams", min_value=1, max_value=20, value=1, step=1)
max_freq_pct = st.sidebar.number_input(
    "Max Player Frequency (%)", min_value=0, max_value=100, value=100, step=5
)
max_occurrences = math.floor(max_freq_pct/100 * num_teams)

# Player upload
uploaded_file = st.file_uploader("Upload Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
else:
    df = pd.read_excel(uploaded_file)
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("❌ File must include 'Name' and 'Value'.")
        st.stop()

    st.subheader("📋 Edit Player Data")
    cols = ["Name", "Value"]
    if "Position" in df.columns: cols.append("Position")
    if "Rank FTPS" in df.columns: cols.append("Rank FTPS")
    if "Bracket" in df.columns: cols.append("Bracket")
    edited = st.data_editor(df[cols], use_container_width=True)

    # compute FTPS
    if "Rank FTPS" in edited.columns:
        rank_map = {r: max(0, 150-(r-1)*5) for r in range(1,31)}
        edited["FTPS"] = edited["Rank FTPS"].apply(lambda x: rank_map.get(int(x),0) if pd.notnull(x) else 0)
    else:
        edited["FTPS"] = 0

    # bracket check
    bracket_fail = False
    if use_bracket_constraints and "Bracket" not in edited.columns:
        st.warning("⚠️ Bracket enabled but no 'Bracket' column.")
        bracket_fail = True

    players = edited.to_dict("records")

    # include/exclude toggles
    if "toggle_choices" not in st.session_state:
        st.session_state.toggle_choices = {}
    default_inc = [n for n,v in st.session_state.toggle_choices.items() if v=="✔"]
    default_exc = [n for n,v in st.session_state.toggle_choices.items() if v=="✖"]
    include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"], default=default_inc)
    exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"], default=default_exc)

    # prepare target_values
    target_values = None
    if solver_mode=="Closest FTP Match" and template_file and format_name:
        try:
            prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
            raw = prof.iloc[:,0].dropna().tolist()
            vals = [float(x) for x in raw if isinstance(x,(int,float)) or str(x).replace('.', '',1).isdigit()]
            if len(vals) < team_size:
                st.error(f"❌ Profile has fewer than {team_size} rows.")
            else:
                target_values = vals[:team_size]
        except Exception as e:
            st.error(f"❌ Failed to read profile: {e}")

    # optimize teams
    if st.sidebar.button("🚀 Optimize Teams"):
        frequency = {p["Name"]:0 for p in players}
        all_teams = []

        def add_bracket(prob, x_vars):
            if use_bracket_constraints and not bracket_fail:
                bd = {}
                for p in players:
                    b = p.get("Bracket")
                    if b: bd.setdefault(b,[]).append(x_vars[p["Name"]])
                for lst in bd.values(): prob += lpSum(lst)<=1

        # LP solvers
        if solver_mode in ["Maximize FTPS","Maximize Budget Usage"]:
            prev = []
            for _ in range(num_teams):
                # build primary LP
                prob = LpProblem("opt",LpMaximize)
                x = {p["Name"]:LpVariable(p["Name"],cat="Binary") for p in players}
                # objective
                if solver_mode=="Maximize FTPS":
                    prob += lpSum(x[n]*next(p["FTPS"] for p in players if p["Name"]=
