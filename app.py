if solver_mode == "Closest FTP Match" and target_values:
    available_players = [p for p in players if p["Name"] not in exclude_players]
    selected_team = []
    used_names = set()
    running_value_total = 0.0

    for target in target_values:
        # Sort players by how close their Value is to the target
        candidates = sorted(
            [p for p in available_players if p["Name"] not in used_names],
            key=lambda p: abs(p["Value"] - target)
        )

        found = False
        for p in candidates[:10]:  # Try top 10 closest options
            if include_players and len(selected_team) < len(include_players) and p["Name"] not in include_players:
                continue
            if running_value_total + p["Value"] <= budget:
                selected_team.append(p)
                used_names.add(p["Name"])
                running_value_total += p["Value"]
                found = True
                break  # Stop at the first valid pick

        if not found:
            st.warning(f"⚠️ No available rider found near value {round(target, 2)} without exceeding budget.")
            break

    if len(selected_team) == team_size:
        result_df = pd.DataFrame(selected_team)
        st.subheader("🎯 Closest Match by Value (Greedy + Budget-Aware)")
        st.dataframe(result_df)
        st.write(f"**Total Value**: {round(running_value_total, 2)}")
        st.write(f"**Total FTPS**: {round(sum(p['FTPS'] for p in selected_team), 2)}")
        st.download_button("📥 Download Team as CSV", result_df.to_csv(index=False), file_name="closest_match_by_value.csv")
    else:
        st.error("❌ Could not select a complete team without exceeding budget.")
