import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random, re, math
from io import BytesIO

st.set_page_config(page_title="Fantasy Team Optimizer v2", layout="wide")
st.title("Fantasy Team Optimizer v2")

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
        if k != "selected_sport":
            del st.session_state[k]
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
budget            = st.sidebar.number_input("Max Budget", value=140.0)
default_team_size = 13
if format_name:
    m = re.search(r"\((\d+)\)", format_name)
    if m:
        default_team_size = int(m.group(1))
team_size = st.sidebar.number_input("Team Size", min_value=1, value=default_team_size, step=1)
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

# === Rank TIERS controls ===
st.sidebar.markdown("### Rank TIERS (Rank Buckets)")
use_tiers = st.sidebar.checkbox(
    "Use Rank Tiers (shuffle FTPS within rank buckets)", value=False,
    help="Players swap FTPS within their rank bucket (e.g., ranks 1–5, 6–10…)."
)
tiers_apply_only_after_first = st.sidebar.checkbox(
    "Apply rank-tier shuffling for teams 2…N only", value=True
)
tiers_text_default = "1-5\n6-10\n11-15\n16-20\n21-25\n26-30"
tiers_text = st.sidebar.text_area(
    "Tier ranges (one per line, e.g., 1-5)", value=tiers_text_default,
    help="Each line is a closed range of Rank positions included in a tier."
)

def parse_tier_ranges(text):
    ranges = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", line)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi:
                ranges.append((lo, hi))
    ranges.sort(key=lambda x: (x[0], x[1]))
    return ranges

tier_ranges = parse_tier_ranges(tiers_text) if use_tiers else []

# --- Global usage cap ---
global_usage_pct = st.sidebar.slider(
    "Global Max Usage % per player (across all teams)",
    0, 100, 100, 5,
    help="Max fraction of teams any player can appear in (INCLUDE still forces 100%)."
)

# === Outcome Tiers (1v1 outcomes) ===
st.sidebar.markdown("### Outcome Tiers (1v1)")
use_outcome_tiers = st.sidebar.checkbox(
    "Use Outcome Tiers (Win/Draw/Loss)", value=False,
    help="Map each player to outcome tiers (1: always win, 2: win/lose, 3: win/draw/lose) and adjust FTPS."
)

allow_draws = st.sidebar.checkbox(
    "Allow Draw outcomes", value=(sport in ["Football", "Soccer", "American Football"])
)

outcome_tier_source = st.sidebar.selectbox(
    "Outcome tier source",
    ["Column 'OutcomeTier' in upload", "Map existing 'Tier' to 1/2/3 if present", "Default everyone to Tier 3"],
    index=1
)

# FTPS multipliers
win_mult  = st.sidebar.number_input("FTPS factor for WIN", 0.0, 10.0, 1.00, 0.05)
draw_mult = st.sidebar.number_input("FTPS factor for DRAW", 0.0, 10.0, 0.60, 0.05, disabled=not allow_draws)
loss_mult = st.sidebar.number_input("FTPS factor for LOSS", 0.0, 10.0, 0.20, 0.05)

# Outcome probabilities
st.sidebar.caption("**Tier 1**: always WIN.")
t2_p_win = st.sidebar.slider("Tier 2: P(Win)", 0.0, 1.0, 0.60, 0.05)
t3_p_win  = st.sidebar.slider("Tier 3: P(Win)",  0.0, 1.0, 0.40, 0.05)
t3_p_draw = st.sidebar.slider("Tier 3: P(Draw)", 0.0, 1.0, 0.25, 0.05, disabled=not allow_draws)

outcome_apply_only_after_first = st.sidebar.checkbox(
    "Apply outcome randomness for teams 2…N only", value=False
)

# --- Rank-1 controls (optional) ---
always_include_rank1_team1 = st.sidebar.checkbox(
    "Always include Rank 1 in Team 1", value=True,
    help="For Team 1 only, force the top-ranked player into the lineup (if feasible)."
)
min_usage_rank1_pct = st.sidebar.slider(
    "Min usage % for Rank 1 (across all teams)", 0, 100, 0, 5,
    help="Ensure the Rank 1 player appears at least this fraction of teams (0% to disable)."
)

# --- Upload & Edit Players ---
st.sidebar.markdown("### Upload Players File")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (players)", type=["xlsx"])
if not uploaded_file:
    st.info("Upload your players file to continue.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Failed to read players file: {e}")
    st.stop()

if not {"Name", "Value"}.issubset(df.columns):
    st.error("❌ File must include 'Name' and 'Value'.")
    st.stop()

# Ensure FTPS exists
if "FTPS" not in df.columns:
    df["FTPS"] = 0.0

# === Rank handling ===
def ensure_rank_column(dframe):
    if "Rank" not in dframe.columns:
        out = dframe.copy()
        out["Rank"] = None
    else:
        out = dframe.copy()
        out["Rank"] = pd.to_numeric(out["Rank"], errors="coerce")

    tmp = out.copy()
    tmp["__order__"] = -pd.to_numeric(tmp["FTPS"], errors="coerce").fillna(0)
    tmp["__name__"] = tmp["Name"].astype(str)

    n = len(tmp)
    ftps_desc_order = tmp.sort_values(["__order__", "__name__"]).reset_index(drop=True)
    ftps_desc_order["__seq__"] = range(1, n + 1)
    order_map = dict(zip(ftps_desc_order["Name"], ftps_desc_order["__seq__"]))

    valid_mask = pd.to_numeric(tmp["Rank"], errors="coerce").notna()
    provided = tmp.loc[valid_mask, "Rank"].astype(int)

    ranks = []
    used = set(int(r) for r in provided if r > 0)
    next_seq = 1
    for _, row in tmp.iterrows():
        if pd.notna(row["Rank"]) and int(row["Rank"]) > 0 and int(row["Rank"]) not in used:
            r = int(row["Rank"]); used.add(r); ranks.append(r)
        else:
            r = order_map[row["Name"]]
            while r in used:
                next_seq += 1
                r = next_seq
            used.add(r); ranks.append(r)

    tmp["Rank"] = ranks
    tmp = tmp.drop(columns=["__order__", "__name__"], errors="ignore")
    return tmp

df = ensure_rank_column(df)

# Compute Rank Tier label per player (e.g. "1-5") if enabled
def tier_label_from_rank(rnk, ranges):
    for lo, hi in ranges:
        if lo <= int(rnk) <= hi:
            return f"{lo}-{hi}"
    return None

if use_tiers and tier_ranges:
    df["Tier"] = df["Rank"].apply(lambda r: tier_label_from_rank(r, tier_ranges))
else:
    if "Tier" not in df.columns:
        df["Tier"] = None

# Editable table
st.subheader("📋 Edit Player Data")
cols = ["Name", "Value"] + [c for c in ("Position", "FTPS", "Bracket", "Rank", "Tier", "OutcomeTier") if c in df.columns]
edited = st.data_editor(df[cols], use_container_width=True)

# Make sure we have FTPS and base copy
if "FTPS" not in edited.columns:
    edited["FTPS"] = 0.0
edited["base_FTPS"] = edited["FTPS"]

# Keep Rank Tier label consistent if user changed ranks
if use_tiers and tier_ranges:
    edited["Tier"] = edited["Rank"].apply(lambda r: tier_label_from_rank(r, tier_ranges))

# --- OutcomeTier derivation ---
def infer_outcome_tier_row(row):
    if outcome_tier_source == "Column 'OutcomeTier' in upload" and "OutcomeTier" in edited.columns:
        v = pd.to_numeric(row.get("OutcomeTier"), errors="coerce")
        return int(v) if pd.notna(v) and v in [1,2,3] else 3
    if outcome_tier_source == "Map existing 'Tier' to 1/2/3 if present":
        raw = row.get("Tier")
        if pd.notna(raw):
            try:
                num = int(raw)
                if num in (1,2,3): 
                    return num
            except:
                s = str(raw)
                if s.startswith("1-"): return 1
                if s.startswith(("6-","2-")): return 2
        return 3
    return 3

if use_outcome_tiers:
    edited["OutcomeTier"] = edited.apply(infer_outcome_tier_row, axis=1).clip(1,3)
else:
    if "OutcomeTier" not in edited.columns:
        edited["OutcomeTier"] = None

players = edited.to_dict("records")
include_players = st.sidebar.multiselect("Players to INCLUDE", edited["Name"])
exclude_players = st.sidebar.multiselect("Players to EXCLUDE", edited["Name"])

# collect brackets
brackets = sorted(edited["Bracket"].dropna().unique()) if "Bracket" in edited.columns else []
if use_bracket_constraints and not brackets:
    st.sidebar.warning("⚠️ Bracket Constraints enabled but no ‘Bracket’ column found.")

# --- Per-Bracket Min/Max count sliders ---
bracket_min_count, bracket_max_count = {}, {}
if brackets:
    with st.sidebar.expander("Usage Count by Bracket", expanded=False):
        for b in brackets:
            bracket_min_count[b] = st.number_input(
                f"Bracket {b} Min picks per team", 0, team_size, 0, 1, key=f"min_{b}"
            )
            bracket_max_count[b] = st.number_input(
                f"Bracket {b} Max picks per team", 0, team_size, team_size, 1, key=f"max_{b}"
            )

# --- Read target profile for Closest FTP Match ---
target_values = None
if solver_mode == "Closest FTP Match" and template_file and format_name:
    try:
        prof = pd.read_excel(template_file, sheet_name=format_name, header=None)
        raw = prof.iloc[:, 0].dropna().tolist()
        vals = [
            float(x) for x in raw
            if isinstance(x, (int, float)) or str(x).replace(".", "", 1).isdigit()
        ]
        if len(vals) < team_size:
            st.error(f"❌ Profile has fewer than {team_size} rows.")
            st.stop()
        target_values = vals[:team_size]
    except Exception as e:
        st.error(f"❌ Failed to read profile: {e}")
        st.stop()

# --- Constraint helpers ---
def add_bracket_constraints(prob, x):
    if use_bracket_constraints:
        for b in brackets:
            members = [x[p["Name"]] for p in players if p.get("Bracket") == b]
            prob += lpSum(members) <= 1, f"UniqueBracket_{b}"

def add_composition_constraints(prob, x):
    for b in brackets:
        mn = bracket_min_count.get(b, 0)
        mx = bracket_max_count.get(b, team_size)
        members = [x[p["Name"]] for p in players if p.get("Bracket") == b]
        if mn > 0:
            prob += lpSum(members) >= mn, f"MinBracket_{b}"
        if mx < team_size:
            prob += lpSum(members) <= mx, f"MaxBracket_{b}"

def add_global_usage_cap(prob, x, prev_sets_local):
    if num_teams <= 1:
        return
    cap = math.floor(num_teams * global_usage_pct / 100)
    for p in players:
        nm = p["Name"]
        if nm in include_players:
            continue
        used = sum(1 for prev in prev_sets_local if nm in prev)
        prob += (used + x[nm] <= cap, f"GlobalUse_{nm}")

def add_min_diff(prob, x, prev_sets_local):
    for idx, prev in enumerate(prev_sets_local):
        prob += lpSum(x[n] for n in prev) <= team_size - diff_count, f"MinDiff_{idx}"

# --- Outcome tier helpers ---
def sample_outcome_for_tier(tier: int):
    if tier == 1:
        return 'W'
    if tier == 2:
        p_win = max(0.0, min(1.0, t2_p_win))
        return 'W' if random.random() < p_win else 'L'
    # tier 3
    pW = max(0.0, min(1.0, t3_p_win))
    pD = 0.0 if not allow_draws else max(0.0, min(1.0 - pW, t3_p_draw))
    r = random.random()
    if r < pW: return 'W'
    if r < pW + pD: return 'D'
    return 'L'

def outcome_factor(symbol: str):
    if symbol == 'W': return win_mult
    if symbol == 'D': return draw_mult if allow_draws else 0.0
    return loss_mult

# --- Build FTPS per team (Outcome tiers -> Rank tiers -> Noise) ---
def build_ftps_values_for_team(team_index: int):
    ftps_vals = {p["Name"]: p.get("base_FTPS", p.get("FTPS", 0.0)) for p in players}

    # A) Outcome tiers (if enabled)
    apply_outcomes_now = use_outcome_tiers and (
        (team_index == 0 and not outcome_apply_only_after_first) or
        (team_index > 0)
    )
    sampled_outcomes = {}
    if apply_outcomes_now:
        for p in players:
            tier = int(p.get("OutcomeTier", 3))
            outcome = sample_outcome_for_tier(tier)
            sampled_outcomes[p["Name"]] = outcome
            ftps_vals[p["Name"]] = ftps_vals[p["Name"]] * outcome_factor(outcome)

    # B) Rank-tier shuffling (if enabled)
    apply_rank_tiers_now = use_tiers and tier_ranges and (
        ((team_index > 0) and tiers_apply_only_after_first) or
        ((team_index == 0) and not tiers_apply_only_after_first) or
        ((team_index > 0) and not tiers_apply_only_after_first)
    )
    if use_outcome_tiers:
        # If you prefer outcome tiers to be exclusive, set apply_rank_tiers_now = False
        pass
    if apply_rank_tiers_now:
        tier_to_players = {}
        for p in players:
            t = p.get("Tier")
            if t:
                tier_to_players.setdefault(t, []).append(p["Name"])
        for t, names in tier_to_players.items():
            if len(names) >= 2:
                values = [ftps_vals[n] for n in names]
                random.shuffle(values)
                for n, v in zip(names, values):
                    ftps_vals[n] = v

    # C) FTPS noise for teams > 1
    if team_index > 0 and ftps_rand_pct > 0:
        for n in ftps_vals:
            ftps_vals[n] = ftps_vals[n] * (1 + random.uniform(-ftps_rand_pct/100, ftps_rand_pct/100))

    return ftps_vals, sampled_outcomes

# --- Optimize Teams ---
if st.sidebar.button("🚀 Optimize Teams"):
    all_teams = []
    prev_sets  = []

    # Helper: get name of Rank 1 player
    def get_rank1_name(ftps_vals=None):
        rank1_candidates = [p for p in players if str(p.get("Rank")) == "1"]
        if rank1_candidates:
            return rank1_candidates[0]["Name"]
        if ftps_vals:
            return max(ftps_vals.items(), key=lambda kv: kv[1])[0]
        # fallback by base FTPS
        return max(players, key=lambda p: p.get("base_FTPS", p.get("FTPS", 0.0)))["Name"]

    # --- Maximize Budget Usage ---
    if solver_mode == "Maximize Budget Usage":
        upper = budget
        for _ in range(num_teams):
            prob = LpProblem("opt_budget", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            prob += lpSum(x[n] * next(q["Value"] for q in players if q["Name"] == n) for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * next(q["Value"] for q in players if q["Name"] == n) for n in x) <= upper

            add_bracket_constraints(prob, x)
            add_composition_constraints(prob, x)
            add_global_usage_cap(prob, x, prev_sets)
            add_min_diff(prob, x, prev_sets)

            for n in include_players:
                prob += x[n] == 1
            for n in exclude_players:
                prob += x[n] == 0

            prob.solve()
            if prob.status != 1:
                st.error("🚫 Infeasible under those constraints.")
                st.stop()

            team = [p for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
            upper = sum(p["Value"] for p in team) - 0.001

    # --- Maximize FTPS ---
    elif solver_mode == "Maximize FTPS":
        for idx in range(num_teams):
            ftps_vals, sampled_outcomes = build_ftps_values_for_team(idx)

            prob = LpProblem(f"opt_ftps_{idx}", LpMaximize)
            x    = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            prob += lpSum(x[n] * ftps_vals[n] for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(x[n] * next(q["Value"] for q in players if q["Name"] == n) for n in x) <= budget

            add_bracket_constraints(prob, x)
            add_composition_constraints(prob, x)
            add_global_usage_cap(prob, x, prev_sets)
            add_min_diff(prob, x, prev_sets)

            # Force includes/excludes
            for n in include_players:
                prob += x[n] == 1
            for n in exclude_players:
                prob += x[n] == 0

            # Rank-1 controls
            rank1_name = get_rank1_name(ftps_vals)
            if idx == 0 and always_include_rank1_team1 and rank1_name in x:
                prob += x[rank1_name] == 1, "ForceRank1Team1"

            if min_usage_rank1_pct > 0 and rank1_name in x:
                min_needed = math.ceil(num_teams * min_usage_rank1_pct / 100)
                used_so_far = sum(1 for prev in prev_sets if rank1_name in prev)
                teams_left = num_teams - idx
                # If we must include Rank 1 now to still hit the minimum, force it
                if used_so_far + teams_left <= min_needed:
                    prob += x[rank1_name] == 1, f"MinUseRank1_team{idx+1}"

            prob.solve()
            if prob.status != 1:
                st.error("🚫 Infeasible under those constraints.")
                st.stop()

            team = []
            for p in players:
                if x[p["Name"]].value() == 1:
                    row = {**p}
                    row["Adjusted FTPS"] = ftps_vals[p["Name"]]
                    # Display outcome info when used
                    if use_outcome_tiers:
                        sym = sampled_outcomes.get(p["Name"], None)
                        row["OutcomeTier"] = p.get("OutcomeTier", 3)
                        row["Outcome"] = sym
                        row["Outcome Factor"] = outcome_factor(sym) if sym else None
                    row["Tier"] = p.get("Tier")
                    team.append(row)

            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})

    # --- Closest FTP Match ---
    else:
        cap = math.floor(num_teams * global_usage_pct / 100)
        for _ in range(num_teams):
            slots, used_brackets, used_names = [None]*team_size, set(), set()

            # place includes
            for n in include_players:
                p0 = next(p for p in players if p["Name"] == n)
                diffs = [(i, abs(p0["Value"] - target_values[i])) for i in range(team_size) if slots[i] is None]
                best_i = min(diffs, key=lambda x: x[1])[0]
                slots[best_i] = p0
                used_names.add(n)
                if use_bracket_constraints and p0.get("Bracket"):
                    used_brackets.add(p0["Bracket"])

            # greedy fill
            for i in range(team_size):
                if slots[i] is not None:
                    continue
                tgt = target_values[i]
                cands = []
                for p in players:
                    if p["Name"] in used_names or p["Name"] in exclude_players:
                        continue
                    if use_bracket_constraints and p.get("Bracket") in used_brackets:
                        continue
                    used = sum(1 for prev in prev_sets if p["Name"] in prev)
                    if p["Name"] not in include_players and used >= cap:
                        continue
                    cands.append(p)
                if not cands:
                    st.error("🚫 Infeasible under those constraints.")
                    st.stop()
                pick = min(cands, key=lambda p: abs(p["Value"] - tgt))
                slots[i] = pick
                used_names.add(pick["Name"])
                if use_bracket_constraints and pick.get("Bracket"):
                    used_brackets.add(pick["Bracket"])

            cost = sum(p["Value"] for p in slots if p)
            if cost > budget:
                st.error(f"❌ Budget exceeded ({cost:.2f} > {budget:.2f}).")
                st.stop()

            current = {p["Name"] for p in slots if p}
            if prev_sets and len(current & prev_sets[-1]) > team_size - diff_count:
                st.error("🚫 Violation of min-difference constraint.")
                st.stop()

            team = []
            for p in slots:
                if p:
                    row = {**p}
                    # No FTPS optimization here; still show derived fields for transparency
                    row["Adjusted FTPS"] = p.get("base_FTPS", p.get("FTPS", 0.0))
                    if use_outcome_tiers:
                        row["OutcomeTier"] = p.get("OutcomeTier", 3)
                    row["Tier"] = p.get("Tier")
                    team.append(row)

            all_teams.append(team)
            prev_sets.append(current)
            if len(all_teams) == num_teams:
                break

    # --- Display each team ---
    for i, team in enumerate(all_teams, start=1):
        with st.expander(f"Team {i}", expanded=(i==1)):
            df_t = pd.DataFrame(team)
            df_t["Selectie (%)"] = df_t["Name"].apply(
                lambda n: round(
                    sum(1 for t in all_teams if any(p["Name"] == n for p in t))
                    / len(all_teams) * 100, 1
                )
            )
            display_cols = [c for c in [
                "Name","Position","Value","Rank","Tier",
                "OutcomeTier","Outcome","Outcome Factor",
                "base_FTPS","Adjusted FTPS","Bracket","Selectie (%)"
            ] if c in df_t.columns]
            display_cols += [c for c in df_t.columns if c not in display_cols]
            st.dataframe(df_t[display_cols], use_container_width=True)

    # --- Build merged DataFrame for download ---
    merged = []
    for idx, team in enumerate(all_teams, start=1):
        df_t = pd.DataFrame(team)
        df_t["Team"] = idx
        df_t["Selectie (%)"] = df_t["Name"].apply(
            lambda n: round(
                sum(1 for t in all_teams if any(p["Name"] == n for p in t))
                / len(all_teams) * 100, 1
            )
        )
        merged.append(df_t)
    merged_df = pd.concat(merged, ignore_index=True)

    # --- Download button (Excel) ---
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        merged_df.to_excel(writer, index=False, sheet_name="All Teams")
    buf.seek(0)

    st.download_button(
        "📥 Download All Teams (Excel)",
        buf,
        file_name="all_teams_v2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
