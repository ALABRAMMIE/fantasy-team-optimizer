    # Maximize FTPS
    elif solver_mode == "Maximize FTPS":
        for idx in range(num_teams):
            # ─── special‐case: single team, no randomness, include is redundant ───
            if (
                num_teams == 1
                and ftps_rand_pct == 0
                and include_players
                and idx == 0
            ):
                # 1) Solve once WITHOUT the include constraint
                prob0 = LpProblem("opt0", LpMaximize)
                x0 = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}
                prob0 += lpSum(x0[n] * p["FTPS"] for p in players for n in [p["Name"]])
                prob0 += lpSum(x0.values()) == team_size
                prob0 += lpSum(
                    x0[n] * next(p["Value"] for p in players if p["Name"] == n)
                    for n in x0
                ) <= budget
                add_bracket_constraints(prob0, x0)
                add_usage_constraints(prob0, x0)
                for n in exclude_players:
                    prob0 += x0[n] == 0
                prob0.solve()
                team0 = {n for n, v in x0.items() if v.value() == 1}

                # 2) if all includes already in that team, just use it
                if set(include_players).issubset(team0):
                    team = [p for p in players if p["Name"] in team0]
                    all_teams.append(team)
                    prev_sets.append(team0)
                    break  # done for the one team

            # ─── otherwise do the normal, full solve ───
            prob = LpProblem("opt", LpMaximize)
            x = {p["Name"]: LpVariable(p["Name"], cat="Binary") for p in players}

            # apply randomness to FTPS after the first team
            if idx > 0 and ftps_rand_pct > 0:
                ftps_values = {
                    p["Name"]: p["FTPS"] * (
                        1 + random.uniform(-ftps_rand_pct/100, ftps_rand_pct/100)
                    )
                    for p in players
                }
            else:
                ftps_values = {p["Name"]: p["FTPS"] for p in players}

            prob += lpSum(x[n] * ftps_values[n] for n in x)
            prob += lpSum(x.values()) == team_size
            prob += lpSum(
                x[n] * next(p["Value"] for p in players if p["Name"] == n)
                for n in x
            ) <= budget

            add_bracket_constraints(prob, x)
            add_usage_constraints(prob, x)

            for n in include_players:
                prob += x[n] == 1
            for n in exclude_players:
                prob += x[n] == 0
            for prev in prev_sets:
                prob += lpSum(x[n] for n in prev) <= team_size - diff_count

            prob.solve()
            if prob.status != 1:
                st.warning("⚠️ LP infeasible for Maximize FTPS.")
                st.stop()

            team = [p for p in players if x[p["Name"]].value() == 1]
            all_teams.append(team)
            prev_sets.append({p["Name"] for p in team})
