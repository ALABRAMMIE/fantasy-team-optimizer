
import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum
import random
import re

st.title("Fantasy Team Optimizer")

sport_options = [
    "-- Choose a sport --",
    "Cycling", "Speed Skating", "Formula 1", "Stock Exchange", "Tennis", "MotoGP", "Football",
    "Darts", "Cyclocross", "Golf", "Snooker", "Olympics", "Basketball", "Dakar Rally", "Skiing",
    "Rugby", "Biathlon", "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment", "Athletics"
]

sport = st.sidebar.selectbox("Select a sport", sport_options)

# Reset session state if sport changes
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for key in list(st.session_state.keys()):
        if key not in ["selected_sport"]:
            del st.session_state[key]
    st.session_state.selected_sport = sport

# Upload template once, early
st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader("Upload Target Profile Template (multi-sheet)", type=["xlsx"], key="template_upload_key")

# Detect formats after sport selection and file upload
available_formats = []
format_name = None
if template_file:
    try:
        xl = pd.ExcelFile(template_file)
        available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
        if available_formats:
            format_name = st.sidebar.selectbox("Select Format", available_formats)
    except:
        st.sidebar.warning("Unable to read sheets from the uploaded template.")

# Bracket constraints toggle
use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")

# Budget & team size
budget = st.sidebar.number_input("Max Budget", value=140.0)

# Default team size logic
default_team_size = 13
if format_name:
    match = re.search(r"\((\d+)\)", format_name)
    if match:
        default_team_size = int(match.group(1))

team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)

solver_mode = st.sidebar.radio("Solver Objective", [
    "Maximize FTPS",
    "Maximize Budget Usage",
    "Closest FTP Match"
])

uploaded_file = st.file_uploader("Upload your Excel file (players)", type=["xlsx"])

if uploaded_file:
    bracket_constraint_failed = False
    df = pd.read_excel(uploaded_file)
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("Uploaded file must include at least 'Name' and 'Value' columns.")
    else:
        st.subheader("📋 Edit Player Data")
        editable_cols = ["Name", "Value"]
        if "Position" in df.columns:
            editable_cols.append("Position")
        if "Rank FTPS" in df.columns:
            editable_cols.append("Rank FTPS")
        if "Bracket" in df.columns:
            editable_cols.append("Bracket")
        edited_df = st.data_editor(df[editable_cols], use_container_width=True)

        if "FTPS" not in edited_df.columns and "Rank FTPS" in edited_df.columns:
            rank_to_points = {rank: max(0, 150 - (rank - 1) * 5) for rank in range(1, 31)}
            edited_df["FTPS"] = edited_df["Rank FTPS"].apply(
                lambda x: rank_to_points.get(int(x), 0) if pd.notnull(x) else 0
            )
        elif "FTPS" not in edited_df.columns:
            edited_df["FTPS"] = 0

        if use_bracket_constraints and "Bracket" not in edited_df.columns:
            st.warning("Bracket constraints are enabled, but no 'Bracket' column found in your data.")
            bracket_constraint_failed = True

        players = edited_df.to_dict("records")

        if "toggle_choices" not in st.session_state:
            st.session_state.toggle_choices = {}

        default_includes = [name for name, val in st.session_state.toggle_choices.items() if val == "✔ Include"]
        default_excludes = [name for name, val in st.session_state.toggle_choices.items() if val == "✖ Exclude"]

        include_players = st.sidebar.multiselect("Players to INCLUDE", edited_df["Name"], default=default_includes)
        exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited_df["Name"], default=default_excludes)

        optimize_clicked = st.sidebar.button("🚀 Optimize Team")

        target_values = None
        if solver_mode == "Closest FTP Match" and template_file and format_name:
            try:
                profile_sheet = pd.read_excel(template_file, sheet_name=format_name, header=None)
                raw_values = profile_sheet.iloc[:, 0].dropna()
                raw_values = [float(x) for x in raw_values if isinstance(x, (int, float)) or str(x).replace('.', '', 1).isdigit()]
                if len(raw_values) < team_size:
                    st.error(f"Target profile for {format_name} has fewer than {team_size} values.")
                else:
                    target_values = raw_values[:team_size]
            except Exception as e:
                st.error(f"Failed to read profile: {e}")


if optimize_clicked:
    st.info("🟡 Optimize button clicked.")
    result_df = None
    if solver_mode == "Closest FTP Match":
        if not target_values:
            st.warning("⚠️ Target values are not loaded. Please check your format sheet or template.")
        elif bracket_constraint_failed:
            st.warning("⚠️ Bracket constraints are enabled, but bracket column is missing.")
    st.info("🟡 Optimize button clicked.")
    result_df = None
        st.warning("⚠️ Target values are not loaded. Please check your format sheet or template.")
        st.warning("⚠️ Target values are not loaded. Please check your format sheet or template.")
        st.warning("⚠️ Bracket constraints are enabled, but bracket column is missing.")
        st.warning("⚠️ Bracket constraints are enabled, but bracket column is missing.")
    st.info("🟡 Optimize button clicked.")
    result_df = None
        st.warning("⚠️ Target values are not loaded. Please check your format sheet or template.")
    st.warning("⚠️ Target values are not loaded. Please check your format sheet or template.")
        st.warning("⚠️ Bracket constraints are enabled, but bracket column is missing.")
    st.warning("⚠️ Bracket constraints are enabled, but bracket column is missing.")
    st.warning("⚠️ Target values are not loaded. Please check your format sheet or template.")
    st.warning("⚠️ Bracket constraints are enabled, but bracket column is missing.")
    st.warning("⚠️ Target values are not loaded. Please check your format sheet or template.")
    st.warning("⚠️ Bracket constraints are enabled, but bracket column is missing.")
    if solver_mode == "Closest FTP Match" and target_values:
    available_players = [p for p in players if p["Name"] not in exclude_players]
    selected_team, used_names = [], set()
    if use_bracket_constraints and not bracket_constraint_failed:
    bracket_used = set()
    for p in available_players:
    if "Bracket" in p and p["Name"] not in used_names:
    if p["Bracket"] in bracket_used:
    continue
    bracket_used.add(p["Bracket"])
    selected_team.append(p)
    used_names.add(p["Name"])
    for target in target_values[len(selected_team):]:
    candidates = sorted(
    [p for p in available_players if p["Name"] not in used_names],
    key=lambda p: abs(p["Value"] - target)
    )
    for p in candidates:
    if p["Name"] not in used_names and (not use_bracket_constraints or p["Bracket"] not in bracket_used):
    selected_team.append(p)
    used_names.add(p["Name"])
    if use_bracket_constraints:
    bracket_used.add(p["Bracket"])
    break
    else:
    for name in include_players:
    force = [p for p in available_players if p["Name"] == name]
    if force:
    selected_team.append(force[0])
    used_names.add(name)
    for target in target_values[len(selected_team):]:
    candidates = sorted(
    [p for p in available_players if p["Name"] not in used_names],
    key=lambda p: abs(p["Value"] - target)
    )
    for p in candidates:
    if p["Name"] not in used_names:
    selected_team.append(p)
    used_names.add(p["Name"])
    break
    if len(selected_team) < team_size:
    st.error(f"❌ Could not select a complete team. Only {len(selected_team)} players chosen.")
else:
    result_df = pd.DataFrame(selected_team)
    st.session_state["result_df"] = result_df

                st.session_state["result_df"] = result_df

        if "result_df" in st.session_state:
            result_df = st.session_state["result_df"]
            st.subheader("🎯 Optimized Team")
            toggle_column = []
            for _, row in result_df.iterrows():
                default = st.session_state.toggle_choices.get(row["Name"], "– Neutral")
                choice = st.radio(
                    f"{row['Name']}", ["✔ Include", "✖ Exclude", "– Neutral"],
                    horizontal=True,
                    key=f"toggle_{row['Name']}",
                    index=["✔ Include", "✖ Exclude", "– Neutral"].index(default)
                )
                st.session_state.toggle_choices[row["Name"]] = choice
                toggle_column.append(choice.split(" ")[0])

            if "🔧" in result_df.columns:
                result_df.drop(columns=["🔧"], inplace=True)
            result_df.insert(0, "🔧", toggle_column)
            st.dataframe(result_df)
