# Write a PGLib-OPF case as PowerModels processes it for rosetta-opf, so that Scaly's model reads exactly
# the numbers JuMP's and ExaModels' models read.
#
#   julia --project=<env> export_case.jl <case.m> <out.json>
#
# The processing is rosetta-opf's (jump.jl, examodels.jl): parse_file (per-unit, corrected angle limits,
# taps, branch directions), standardize_cost_terms!(order=2), calc_thermal_limits!, build_ref. Every
# collection is written in the order the JuMP model iterates it (Julia's Dict order), and each branch
# carries the admittance and tap terms of the flow equations (calc_branch_y, calc_branch_t).

import PowerModels, JSON
PowerModels.silence()

function export_case(file, out)
    data = PowerModels.parse_file(file)
    PowerModels.standardize_cost_terms!(data, order=2)
    PowerModels.calc_thermal_limits!(data)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    buses = [begin
        loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]
        Dict("id" => i, "vmin" => b["vmin"], "vmax" => b["vmax"],
             "pd" => sum(l["pd"] for l in loads; init=0.0), "qd" => sum(l["qd"] for l in loads; init=0.0),
             "gs" => sum(s["gs"] for s in shunts; init=0.0), "bs" => sum(s["bs"] for s in shunts; init=0.0))
    end for (i, b) in ref[:bus]]
    gens = [Dict("id" => i, "bus" => g["gen_bus"], "pmin" => g["pmin"], "pmax" => g["pmax"], "qmin" => g["qmin"],
                 "qmax" => g["qmax"], "cost" => g["cost"]) for (i, g) in ref[:gen]]
    branches = [begin
        g, b = PowerModels.calc_branch_y(br)
        tr, ti = PowerModels.calc_branch_t(br)
        Dict("id" => i, "f_bus" => br["f_bus"], "t_bus" => br["t_bus"], "g" => g, "b" => b, "tr" => tr, "ti" => ti,
             "g_fr" => br["g_fr"], "b_fr" => br["b_fr"], "g_to" => br["g_to"], "b_to" => br["b_to"],
             "angmin" => br["angmin"], "angmax" => br["angmax"], "rate_a" => br["rate_a"])
    end for (i, br) in ref[:branch]]
    arcs = [[l, i, j] for (l, i, j) in ref[:arcs]]
    open(out, "w") do f
        JSON.print(f, Dict("case" => basename(file), "baseMVA" => data["baseMVA"], "bus" => buses, "gen" => gens,
                           "branch" => branches, "arcs" => arcs, "ref_buses" => collect(keys(ref[:ref_buses]))))
    end
end

export_case(ARGS[1], ARGS[2])
