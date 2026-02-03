import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random
from io import BytesIO
from collections import defaultdict

# --- Page Config ---
st.set_page_config(page_title="Team Optimizer v8 (Max Mode)", layout="wide")
st.title("⚽/🏀 Team Optimizer v8 (Auto-Max Mode)")
st.markdown("""
**Nieuw in v8:**
* **Auto-Stop:** Vink 'Stop als op' aan. De optimizer berekent zoveel teams als wiskundig mogelijk is.
* **Rank 1 Fix:** Zet 'Forceer Rank 1' uit als je vastloopt op budget.
""")

# ==========================================
# 1. SIDEBAR INSTELLINGEN
# ==========================================
st.sidebar.header("⚙️ Solver Settings")

budget = st.sidebar.number_input("Max Budget", value=58.0, step=0.5)
team_size = st.sidebar.number_input("Team Grootte", min_value=1, value=9)
num_lineups = st.sidebar.number_input("Aantal Lineups (Doel)", min_value=1, max_value=100, value=20)
min_diff = st.sidebar.number_input("Minimaal verschil (spelers)", value=1)

# Constraints
st.sidebar.markdown("---")
st.sidebar.markdown("**Constraints**")
avoid_opposing = st.sidebar.checkbox("Game/Bracket Constraint (Max 1 per groep)", value=True)
include_rank1 = st.sidebar.checkbox("⭐ Forceer Rank 1 speler in Team 1", value=False, help="Zet dit UIT als je budget problemen krijgt!")
stop_early = st.sidebar.checkbox("⏹️ Stop als op (Maximaliseer)", value=True, help="Stopt netjes als er geen unieke teams meer gemaakt kunnen worden.")

st.sidebar.markdown("---")
st.sidebar.header("🎲 Tiers Config")

# Tiers Tabel
default_data = {
    "Tier": [1, 2, 3, 4, 5],
    "Label": ["Heavy Fav", "Favorite", "Toss Up", "Underdog", "Longshot"],
    "Win %": [90, 70, 50, 30, 10],      
    "Pts WIN": [450, 450, 450, 450, 450], 
    "Pts LOSS": [200, 180, 150, 100, 50] 
}

edited_tiers = st.sidebar.data_editor(
    pd.DataFrame(default_data),
    hide_index=True,
    num_rows="dynamic",
    key="tiers_v8"
)

# Settings laden
tier_settings = {}
for i, row in edited_tiers.iterrows():
    try:
        tier_settings[int(row["Tier"])] = {
            "prob": row["Win %"],
            "pts_win": row["Pts WIN"],
            "pts_loss": row["Pts LOSS"]
        }
    except: pass

# ==========================================
# 2. UPLOAD & DATA PREVIEW
# ==========================================
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "csv"])

if uploaded_file:
    # Lezen (ondersteunt csv en xlsx)
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Fout bij lezen bestand: {e}")
        st.stop()
    
    # Kolom check & fix
    # We proberen slim te zijn met kolomnamen
    if "FTPS" not in df.columns and "Value" in df.columns:
        df["FTPS"] = df["Value"] # Fallback
    
    if "OutcomeTier" not in df.columns:
        df["OutcomeTier"] = 3 # Default naar tier 3
    
    # GameID logic: Gebruik 'Bracket' als 'GameID' mist
    if "GameID" not in df.columns:
        if "Bracket" in df.columns:
            df["GameID"] = df["Bracket"]
        else:
            df["GameID"] = df.index # Geen koppeling
    
    # Types
    df["OutcomeTier"] = pd.to_numeric(df["OutcomeTier"], errors='coerce').fillna(3).astype(int)
    # Zorg dat Value een float is
    df["Value"] = pd.to_numeric(df["Value"], errors='coerce').fillna(0)
    
    players_data = df.to_dict("records")
    
    with st.expander("🔍 Bekijk geüploade data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Forceer opties
    all_names = sorted(df["Name"].unique().astype(str))
    st.sidebar.markdown("---")
    st.sidebar.header("🔒 Forceer Teams")
    must_include = st.sidebar.multiselect("Moet in team (Include):", all_names)
    must_exclude = st.sidebar.multiselect("Mag niet in team (Exclude):", [n for n in all_names if n not in must_include])

    # Rank 1 bepalen (voor de constraint)
    rank1_player = max(players_data, key=lambda x: x.get("FTPS", 0))

    # ==========================================
    # 3. SIMULATIE LOGICA
    # ==========================================
    def run_matchups(teams_data):
        sim_scores = {}
        sim_outcomes = {}
        
        games = defaultdict(list)
        for t in teams_data:
            # Gebruik string conversie voor zekerheid
            gid = str(t.get("GameID"))
            games[gid].append(t)

        for gid, opponents in games.items():
            if len(opponents) == 2:
                tA = opponents[0]
                tB = opponents[1]
                
                settA = tier_settings.get(tA["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                settB = tier_settings.get(tB["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                
                weight_A = settA["prob"]
                weight_B = settB["prob"]
                total = weight_A + weight_B
                if total == 0: total = 1
                
                if random.random() < (weight_A / total):
                    sim_scores[tA["Name"]] = settA["pts_win"]
                    sim_outcomes[tA["Name"]] = "WIN"
                    sim_scores[tB["Name"]] = settB["pts_loss"]
                    sim_outcomes[tB["Name"]] = "LOSS"
                else:
                    sim_scores[tA["Name"]] = settA["pts_loss"]
                    sim_outcomes[tA["Name"]] = "LOSS"
                    sim_scores[tB["Name"]] = settB["pts_win"]
                    sim_outcomes[tB["Name"]] = "WIN"
            else:
                for t in opponents:
                    sett = tier_settings.get(t["OutcomeTier"], {"prob": 50, "pts_win":0, "pts_loss":0})
                    is_win = random.random() < (sett["prob"] / 100.0)
                    sim_scores[t["Name"]] = sett["pts_win"] if is_win else sett["pts_loss"]
                    sim_outcomes[t["Name"]] = "WIN" if is_win else "LOSS"
        return sim_scores, sim_outcomes

    # ==========================================
    # 4. OPTIMIZATION LOOP
    # ==========================================
    if st.button("🚀 Start Optimalisatie"):
        
        progress = st.progress(0)
        results = []
        prev_lineups = []
        
        # Budget Check Includes
        inc_cost = sum(t["Value"] for t in players_data if t["Name"] in must_include)
        if inc_cost > budget:
            st.error(f"Includes te duur ({inc_cost:.1f} > {budget})!")
            st.stop()

        status_placeholder = st.empty()

        for i in range(num_lineups):
            status_placeholder.text(f"Bezig met lineup {i+1}...")
            
            # 1. Simuleer
            scores, outcomes = run_matchups(players_data)
            
            # 2. Solver Setup
            prob = LpProblem(f"Lineup_{i}", LpMaximize)
            x = LpVariable.dicts("Select", [str(t["Name"]) for t in players_data], cat="Binary")
            
            # Objective
            prob += lpSum([x[str(t["Name"])] * scores[t["Name"]] for t in players_data])
            
            # Base Constraints
            prob += lpSum([x[str(t["Name"])] for t in players_data]) == team_size
            prob += lpSum([x[str(t["Name"])] * t["Value"] for t in players_data]) <= budget
            
            # Min Diff (Uniqueness)
            for prev in prev_lineups:
                prob += lpSum([x[str(n)] for n in prev]) <= (team_size - min_diff)
            
            # Game/Bracket Constraint
            if avoid_opposing:
                game_map = defaultdict(list)
                for t in players_data:
                    gid = str(t.get("GameID", "UNK"))
                    game_map[gid].append(t["Name"])
                for gid, names in game_map.items():
                    if len(names) > 1:
                        prob += lpSum([x[str(n)] for n in names]) <= 1
            
            # Includes/Excludes
            for n in must_include: prob += x[str(n)] == 1
            for n in must_exclude: prob += x[str(n)] == 0
            
            # Rank 1 Constraint (Optional)
            if include_rank1 and i == 0:
                if rank1_player["Name"] not in must_exclude:
                    prob += x[str(rank1_player["Name"])] == 1

            # 3. Solve
            # We zetten msg=0 om console logs te vermijden
            prob.solve()
            
            # 4. Check Status
            if prob.status == 1:
                selected = [t["Name"] for t in players_data if x[str(t["Name"])].value() == 1]
                prev_lineups.append(set(selected))
                
                for t in players_data:
                    if t["Name"] in selected:
                        row = t.copy()
                        row["Simulated Points"] = scores[t["Name"]]
                        row["Simulated Outcome"] = outcomes[t["Name"]]
                        row["Lineup ID"] = i + 1
                        results.append(row)
            else:
                # OPLOSSING VOOR JOUW PROBLEEM
                if stop_early:
                    st.warning(f"⚠️ Gestopt na {i} lineups. Geen unieke combinaties meer mogelijk met deze constraints.")
                    break
                else:
                    # Als je niet wilt stoppen, kunnen we een error tonen of doorgaan
                    # Maar meestal heeft doorgaan geen zin bij constraints.
                    pass
            
            progress.progress((i+1)/num_lineups)
            
        status_placeholder.empty()

        if results:
            df_res = pd.DataFrame(results)
            st.success(f"Klaar! {len(df_res['Lineup ID'].unique())} lineups gegenereerd.")
            
            tabs = st.tabs(["Per Lineup", "Excel Data"])
            with tabs[0]:
                for lid in sorted(df_res["Lineup ID"].unique()):
                    subset = df_res[df_res["Lineup ID"] == lid]
                    t_pts = subset["Simulated Points"].sum()
                    t_cost = subset["Value"].sum()
                    with st.expander(f"Lineup {lid} (Pts: {t_pts} | Cost: {t_cost:.1f})", expanded=(lid==1)):
                        cols = ["Name", "Value", "Simulated Outcome", "Simulated Points", "GameID", "OutcomeTier"]
                        st.dataframe(subset[[c for c in cols if c in subset.columns]])

            with tabs[1]:
                st.dataframe(df_res)
                
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_res.to_excel(writer, index=False)
            buf.seek(0)
            st.download_button("📥 Download Excel", buf, "optimized_teams.xlsx")
        else:
            st.error("🚫 Geen enkel team gevonden. Zet 'Forceer Rank 1' uit of verhoog je budget iets!")
