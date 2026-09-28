# Run one of rosetta-opf's models, unmodified, on one case, and write what Ipopt reports.
#
#   julia --project=<env> run_rosetta.jl <model file> <case.m> <out.json>
#
# The model file is rosetta-opf's `jump.jl`, or its `examodels.jl` with `examodels_v0.12.patch` (the
# ExaModels 0.12 names for the same calls), in a copy of rosetta-opf's directory. Run from a directory
# holding an `ipopt.opt` (compare.py writes it): Ipopt reads its options from there, so the models' own
# code is used as published. Including the model file runs rosetta-opf's
# warm-up case (the script does so when not interactive), which compiles; then the case is solved twice
# and the second solve is the one recorded (the first compiles whatever the warm-up case did not).
# Ipopt's output is parsed by compare.py; this writes rosetta-opf's own timings.

import JSON

model, case, out = ARGS
t0 = time()
include(model)
t_warmup = time() - t0
println("\n==== SOLVE ====")
t0 = time()
first_result = solve_opf(case)
t_first = time() - t0
println("\n==== SOLVE ====")
t0 = time()
result = solve_opf(case)
t_case = time() - t0
delete!(result, "solution")
result["t_warmup_process"] = t_warmup
result["first_call"] = t_first
result["t_case_call"] = t_case
open(out, "w") do f
    JSON.print(f, result)
end
