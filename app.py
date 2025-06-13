import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
import re
import math
from io import BytesIO

st.title("Fantasy Team Optimizer")

# --- Sidebar Inputs ---
sport_options = [
    "-- Choose a sport --", "Cycling", "Speed Skating", "Formula 1", "Stock Exchange",
    "Tennis", "MotoGP", "Football", "Darts", "Cyclocross", "Golf", "Snooker",
    "Olympics", "Basketball", "Dakar Rally", "Skiing", "Rugby", "Biathlon",
    "Handball", "Cross Country", "Baseball", "Ice Hockey", "American Football",
    "Ski Jumping", "MMA", "Entertainment", "Athletics"
]
sport = st.sidebar.selectbox("Select a sport", sport_options)

# Reset state when sport changes
if "selected_sport" not in st.session_state:
    st.session_state.selected_sport = sport
elif sport != st.session_state.selected_sport:
    for k in list(st.session_state.keys()):
        if k != "selected_sport":
            del st.session_state[k]
    st.session_state.selected_sport = sport

# --- Upload Profile Template (multi-sheet) ---
st.sidebar.markdown("### Upload Profile Template")
template_file = st.sidebar.file_uploader(
    "Upload Target Profile Template (multi-sheet)", type=["xlsx"], key="template_upload_key"
)
available_formats = []
format_name = None
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
budget = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", value=default_team_size, step=1)
solver_mode = st.sidebar.radio(
    "Solver Objective",
    ["Maximize FTPS", "Maximize Budget Usage", "Closest FTP Match"]
)
num_teams = st.sidebar.number_input(
    "Number of Teams", min_value=1, max_value=25, value=1
)
diff_count = st.sidebar.number_input(
    "Min Verschil tussen Teams (aantal spelers)",
    min_value=0,
    max_value=team_size,
    value=1
)

# --- New: max usage % per player/team ---
max_usage_pct = st.sidebar.slider(
    "Max Usage % per player/team",
    min_value=0,
    max_value=100,
    value=100,
    step=5,
    help="Cap the fraction of teams any one player can appear on (excludes forced include/exclude)."
)

# --- New: FTPS randomness % for subsequent teams ---
ftps_rand_pct = st.sidebar.slider(
    "FTPS Randomness % for subsequent teams",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    help="Apply ± this percent random noise to FTPS values for teams 2…N."
)

# --- Upload Players File ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader(
    "Upload your Excel file (players)", type=["xlsx"]
)
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

def load_players(file):
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Failed to read players file: {e}")
        st.stop()
    if not {"Name", "Value"}.issubset(df.columns):
        st.error("❌ File must include 'Name' and 'Value'.")
        st.stop()
    return df

df = load_players(uploaded_file)

# --- Edit player data and snapshot base_FTPS ---
st.
