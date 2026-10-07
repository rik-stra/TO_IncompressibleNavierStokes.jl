# The per-QoI ridge rule of LinReg^E (paper closures.tex, step 1; Rik 2026-10-06): ridge on every QoI
# whose correction has an integral time shorter than the history length h = 5 steps, measured on
# the TRAINING window (1-10 TU, steps 400-4000 of R1's tracked record).
#
#     julia --startup-file=no --project=lib/RikFlow lib/RikFlow/analysis/tint_rule.jl
#
# ⚠️ `results.md` §1's T_int(dQ) table is over the whole 100 TU record (`score_m0_ddn.jl`, `ref_dQ`),
# not the training window; the rule uses this script's numbers. T_int = `correlation_time`'s
# integral of the ACF to its first zero crossing (cap 500 lags; `truncated` = no zero crossing).
using RikFlow, JLD2, Printf
const DT = 2.5e-3
const H = 5
rec = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
dQ = rec["q"][:, 2:end] .- rec["q_star"]          # the correction q^{n+1} - q*^n, as the fit sees it
labels = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
for (w, cols) in (("training window 1-10 TU (steps 400-4000) -- the rule", 400:4000),
                  ("whole record 0-100 TU (results.md Sec. 1, for comparison)", axes(dQ, 2)))
    println(w)
    for i in eachindex(labels)
        c = RikFlow.correlation_time(collect(dQ[i, cols]), DT)
        st = c.T_int / DT
        @printf("  %-9s T_int %.4f TU = %6.1f steps%s%s\n", labels[i], c.T_int, st,
                c.truncated ? " (truncated: lower bound)" : "", st < H ? "  <- below h = 5: ridge" : "")
    end
end
