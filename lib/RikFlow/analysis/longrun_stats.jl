# Long-run statistics of the baselines as the paper defines them (Sec. 3.4, Table "measures"):
# every measure per QoI first, then combined over the six QoIs.
#
#   summed KS D      sum over the QoIs of D_i; per run (range) and the five runs pooled
#   mean bias b      mean over the QoIs of b_i = <x_i>/<y_i> - 1; per run (range) and pooled (the mean
#                    over the runs); also per block of 10 TU, the reference over the same block
#   long-run ratio   Eq. spread-skill with the five runs as members, k over every step, the sums also
#                    over the QoIs, each QoI divided by sigma_i (std of the reference, whole record)
#   stable runs      runs that reach 100 TU with every QoI finite
#
# Rik, 2026-10-12: "define how every measure combines the six QoIs". Replaces the scratch scripts of
# 2026-10-08/12 (perrun_bias.jl, pooled_ks.jl). score_m0_ddn.jl's `ss_of` used to pool the raw sums
# over the QoIs, which the Z bands (about 1e3, against 0.07 for E[16,32]) dominate; it divides by
# sigma_i too since 2026-10-12. The per-QoI ratios are close, so the two poolings differ by at most
# 0.02 except at lambda = 1e4 (0.17 raw, 0.21 here).
#
# Usage: julia --project=analysis analysis/longrun_stats.jl

using JLD2, Statistics, Printf

const HERE = @__DIR__
include(joinpath(HERE, "..", "src", "ts_score.jl"))
const OUTD = joinpath(HERE, "..", "exp_square_HIT", "output")
const QREF = load(joinpath(HERE, "data", "hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2"), "q_ref")
const SIG = vec(std(QREF; dims = 2))
const LABELS = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const NSTEP = 40001                                    # 100 TU at dt = 2.5e-3, both ends
const BLOCKS = [((b - 1) * 4000 + 1):(b * 4000 + 1) for b in 1:10]
const CLOSURES = (("LinReg1", "0"), ("LinReg6", "1e-4"), ("LinReg7", "1"), ("LinReg8", "10"),
                  ("LinReg9", "100"), ("LinReg10", "1e4"), ("DDN", "-"))

function runs(c)
    dir, pat = c == "DDN" ? (joinpath(OUTD, "TO_DDN"), r"^DDN_data_online_tsim100\.0_replica\d\.jld2$") :
                            (joinpath(OUTD, "TO_LRS", c), r"^data_online_tsim100\.0_replica\d\.jld2$")
    fs = sort(filter(f -> occursin(pat, f), readdir(dir)))
    return [jldopen(io -> Float64.(io["data_online"].q), joinpath(dir, f)) for f in fs]
end

stable(r) = size(r, 2) >= NSTEP && all(isfinite, r)

"Mean bias per QoI, in %, of `x` against the reference over the same columns."
bias_i(x, cols) = [100 * (mean(view(x, i, cols)) / mean(view(QREF, i, cols)) - 1) for i in 1:6]

"Long-run spread-skill ratio, the QoIs divided by sigma_i and pooled."
function longrun_ratio(rs)
    K = min(minimum(size(r, 2) for r in rs), size(QREF, 2))
    arr = Array{Float64}(undef, K, 6, length(rs))
    for (j, r) in enumerate(rs)
        arr[:, :, j] .= transpose(view(r, :, 1:K)) ./ SIG'
    end
    return spread_skill(arr, Matrix(transpose(view(QREF, :, 1:K))) ./ SIG')
end

function main()
    @printf("%-9s %-5s %6s %-15s %6s %-12s %6s %s\n", "closure", "lam", "b", "b single runs", "D",
            "D single", "ratio", "stable")
    detail = String[]
    for (c, lam) in CLOSURES
        rs = runs(c)
        K = minimum(size(r, 2) for r in rs)
        B = reduce(hcat, [bias_i(r, 1:K) for r in rs])              # QoIs x runs
        b_run = vec(mean(B; dims = 1))
        D_run = [summed_ks(r, QREF).total for r in rs]
        Dp = ensemble_ks(rs, QREF).total
        ss = longrun_ratio(rs)
        nst = count(stable, rs)
        @printf("%-9s %-5s %+6.1f [%+5.1f, %+5.1f] %6.2f %.2f-%.2f %6.2f %d of %d\n", c, lam,
                mean(b_run), minimum(b_run), maximum(b_run), Dp, minimum(D_run), maximum(D_run),
                ss.ratio, nst, length(rs))
        # sign agreement over the QoIs, per run; the pooled b_i per QoI; the blocks
        mixed = count(j -> minimum(B[:, j]) < 0 < maximum(B[:, j]), axes(B, 2))
        bpool = vec(mean(B; dims = 2))
        blk = [mean(bias_i(r, bl)) for r in rs, bl in BLOCKS]       # runs x blocks, QoI mean
        blkm = vec(mean(blk; dims = 1))
        push!(detail, @sprintf("%-9s runs with b_i of both signs %d; pooled b_i %s; per-QoI ratio %s\n%9s blocks, mean of the runs: %s\n%9s blocks, single runs: %+.1f to %+.1f",
              c, mixed, join((@sprintf("%s %+.1f", LABELS[i], bpool[i]) for i in 1:6), " "),
              join((@sprintf("%.2f", v) for v in ss.per_qoi), " "), "",
              join((@sprintf("%+.1f", v) for v in blkm), " "), "", minimum(blk), maximum(blk)))
    end
    println()
    foreach(println, detail)
end

main()
