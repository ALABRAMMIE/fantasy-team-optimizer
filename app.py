import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import random, re, math
from io import BytesIO

st.title("Fantasy Team Optimizer")

# --- Sidebar: Sport & Template ---
sport_options = [...]
# [rest of imports and sidebar setup unchanged]
# --- (previous code remains identical up to Optimize Teams) ---

if st.sidebar.button("🚀 Optimize Teams"):
    all_teams = []
    prev_sets = []
    # [solver logic populates all_teams and subs]

    # --- Prepare output with roles ---
    output_records = []
    # Main teams
    for idx, team in enumerate(all_teams, start=1):
        for p in team:
            rec = p.copy()
            rec['Team'] = idx
            rec['Role'] = 'Main'
            output_records.append(rec)
    # Substitutes
    if tour_mode and sport == 'Cycling':
        sub_idx = len(all_teams) + 1
        for p in subs:
            rec = p.copy()
            rec['Team'] = sub_idx
            rec['Role'] = 'Substitute'
            output_records.append(rec)

    # --- Display each main team separately ---
    for idx in range(1, len(all_teams) + 1):
        with st.expander(f"Team {idx}"):
            df_t = pd.DataFrame([r for r in output_records if r['Team'] == idx])
            df_t['Selectie (%)'] = df_t['Name'].apply(
                lambda n: round(
                    sum(1 for r in output_records if r['Name'] == n and r['Role'] == 'Main')
                    / len(all_teams) * 100, 1
                )
            )
            st.dataframe(df_t)

    # --- Display substitutes in their own expander ---
    if tour_mode and sport == 'Cycling':
        with st.expander("🔄 Tour Substitutes"):
            df_s = pd.DataFrame([r for r in output_records if r['Role'] == 'Substitute'])
            df_s['Selectie (%)'] = df_s['Name'].apply(
                lambda n: round(
                    sum(1 for r in output_records if r['Name'] == n and r['Role'] == 'Main')
                    / len(all_teams) * 100, 1
                )
            )
            st.dataframe(df_s.style.apply(lambda _: ['background-color: lightyellow'] * len(df_s), axis=1))

    # --- Download combined sheet ---
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        pd.DataFrame(output_records).to_excel(writer, sheet_name="All Teams", index=False)
    buf.seek(0)
    st.download_button(
        "📥 Download All Teams (Excel)",
        buf,
        file_name="all_teams.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
