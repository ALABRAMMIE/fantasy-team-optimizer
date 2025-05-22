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

# Reset all session state when sport changes
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for key in list(st.session_state.keys()):
        if key != "selected_sport":
            del st.session_state[key]
    st.session_state.selected_sport = sport

# Upload profile template (multi-sheet)
st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template (multi-sheet)",
    type=["xlsx"],
    key="template_upload_key"
)

# Detect available formats (sheet names) that match the selected sport
available_formats = []
format_name = None
if template_file:
    try:
        xl = pd.ExcelFile(template_file)
        available_formats = [s for s in xl.sheet_names if s.startswith(sport)]
        if available_formats:
            format_name = st.sidebar.selectbox("Select Format", available_formats)
    except:
        st.sidebar.warning("⚠️ Unable to read sheet names from the template.")

# Optional bracket constraints
use_bracket_constraints = st.sidebar.checkbox("Use Bracket Constraints")

# Budget & team size inputs
budget = st.sidebar.number_input("Max Budget", value=140.0)

# Auto-detect team size from the chosen format (e.g. "Cycling - Day (13)")
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)

# Solver mode
solver_mode = st.sidebar.radio("Solver Objective", [
    "Maximize FTPS",
    "Maximize Budget Usage",
    "Closest FTP Match"
])

# Player file upload
uploaded_file = st.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("❌ Your file must include at least 'Name' and 'Value' columns.")
    else:
        # Allow editing key columns
        st.subheader("📋 Edit Player Data")
        editable_cols = ["Name", "Value"]
        if "Position" in df.columns:
            editable_cols.append("Position")
        if "Rank FTPS" in df.columns:
            editable_cols.append("Rank FTPS")
        if "Bracket" in df.columns:
            editable_cols.append("Bracket")
        edited_df = st.data_editor(df[editable_cols], use_container_width=True)

        # Compute FTPS if Rank FTPS provided
        if "Rank FTPS" in edited_df.columns:
            rank_to_points = {r: max(0, 150 - (r - 1) * 5) for r in range(1, 31)}
            edited_df["FTPS"] = edited_df["Rank FTPS"].apply(
                lambda x: rank_to_points.get(int(x), 0) if pd.notnull(x) else 0
            )
        else:
            edited_df["FTPS"] = 0

        # Check bracket column if constraint enabled
        bracket_constraint_failed = False
        if use_bracket_constraints and "Bracket" not in edited_df.columns:
            st.warning("⚠️ Bracket constraints enabled but no 'Bracket' column found.")
            bracket_constraint_failed = True

        players = edited_df.to_dict("records")

        # Initialize toggle state
        if "toggle_choices" not in st.session_state:
            st.session_state.toggle_choices = {}

        default_includes = [
            name for name, v in st.session_state.toggle_choices.items() if v == "✔"
        ]
        default_excludes = [
            name for name, v in st.session_state.toggle_choices.items() if v == "✖"
        ]
        include_players = st.sidebar.multiselect(
            "Players to INCLUDE", edited_df["Name"], default=default_includes
        )
        exclude_players = st.sidebar.multiselect(
            "Players to EXCLUDE", edited_df["Name"], default=default_excludes
        )

        optimize_clicked = st.sidebar.button("🚀 Optimize Team")

        # Prepare target values for Closest FTP Match
        target_values = None
        if solver_mode == "Closest FTP Match" and template_file and format_name:
            try:
                profile_sheet = pd.read_excel(template_file, sheet_name=format_name, header=None)
                raw = profile_sheet.iloc[:, 0].dropna().tolist()
                vals = [
                    float(x) for x in raw
                    if isinstance(x, (int, float)) or str(x).replace(".", "", 1).isdigit()
                ]
                if len(vals) < team_size:
                    st.error(f"❌ Profile for {format_name} has fewer than {team_size} values.")
                else:
                    target_values = vals[:team_size]
            except Exception as e:
                st.error(f"❌ Failed to read profile: {e}")

        # ==== OPTIMIZATION ====
        if optimize_clicked:
            st.info("🟡 Optimize button clicked.")
            result_df = None

            # Closest FTP Match
            if solver_mode == "Closest FTP Match":
                if not target_values:
                    st.warning("⚠️ Target values not loaded. Check your template & format.")
                elif use_bracket_constraints and bracket_constraint_failed:
                    st.warning("⚠️ Bracket constraint enabled but no column present.")
                else:
                    available = [p for p in players if p["Name"] not in exclude_players]
                    selected_team, used = [], set()

                    # Force includes first
                    for name in include_players:
                        for p in available:
                            if p["Name"] == name and name not in used:
                                selected_team.append(p)
                                used.add(name)
                                break

                    # Bracket-based picks
                    if use_bracket_constraints:
                        bracket_used = set()
                        # pick one per bracket first
                        for p in available:
                            b = p.get("Bracket")
                            if b and p["Name"] not in used and b not in bracket_used:
                                selected_team.append(p)
                                used.add(p["Name"])
                                bracket_used.add(b)
                                if len(selected_team) == team_size:
                                    break
                        # fill remaining by closeness
                        for tgt in target_values[len(selected_team):]:
                            candidates = sorted(
                                [p for p in available if p["Name"] not in used and p.get("Bracket") not in bracket_used],
                                key=lambda p: abs(p["Value"] - tgt)
                            )
                            if candidates:
                                c = candidates[0]
                                selected_team.append(c)
                                used.add(c["Name"])
                                bracket_used.add(c.get("Bracket"))
                    else:
                        # purely by closeness
                        for tgt in target_values[len(selected_team):]:
                            candidates = sorted(
                                [p for p in available if p["Name"] not in used],
                                key=lambda p: abs(p["Value"] - tgt)
                            )
                            if candidates:
                                c = candidates[0]
                                selected_team.append(c)
                                used.add(c["Name"])

                    if len(selected_team) == team_size:
                        result_df = pd.DataFrame(selected_team)
                        st.session_state["result_df"] = result_df
                    else:
                        st.error(f"❌ Only {len(selected_team)} selected. Couldn't form a full team.")

            # Maximize FTPS
            elif solver_mode == "Maximize FTPS":
                prob = LpProblem("MaximizeFTPS", LpMaximize)
                x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
                prob += lpSum(x[p["Name"]] * p.get("FTPS", 0) for p in players)
                prob += lpSum(x[p["Name"]] for p in players) == team_size
                prob += lpSum(x[p["Name"]] * p["Value"] for p in players) <= budget
                for name in include_players:
                    prob += x[name] == 1
                for name in exclude_players:
                    prob += x[name] == 0
                prob.solve()
                selected = [p for p in players if x[p["Name"]].value() == 1]
                result_df = pd.DataFrame(selected)
                st.session_state["result_df"] = result_df

            # Maximize Budget Usage
            elif solver_mode == "Maximize Budget Usage":
                prob = LpProblem("MaximizeBudget", LpMaximize)
                x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
                prob += lpSum(x[p["Name"]] * p["Value"] for p in players)
                prob += lpSum(x[p["Name"]] for p in players) == team_size
                prob += lpSum(x[p["Name"]] * p["Value"] for p in players) <= budget
                for name in include_players:
                    prob += x[name] == 1
                for name in exclude_players:
                    prob += x[name] == 0
                prob.solve()
                selected = [p for p in players if x[p["Name"]].value() == 1]
                result_df = pd.DataFrame(selected)
                st.session_state["result_df"] = result_df

        # ==== DISPLAY RESULT ====
        if "result_df" in st.session_state:
            df_res = st.session_state["result_df"]
            st.subheader("🎯 Optimized Team")

            # Toggles per player
            toggle_col = []
            for _, row in df_res.iterrows():
                default = st.session_state.toggle_choices.get(row["Name"], "–")
                choice = st.radio(
                    row["Name"],
                    ["✔", "✖", "–"],
                    horizontal=True,
                    key=f"tog_{row['Name']}",
                    index=["✔", "✖", "–"].index(default)
                )
                st.session_state.toggle_choices[row["Name"]] = choice
                toggle_col.append(choice)

            # Insert symbol column
            if "🔧" in df_res.columns:
                df_res = df_res.drop(columns=["🔧"])
            df_res.insert(0, "🔧", toggle_col)
            st.dataframe(df_res)
