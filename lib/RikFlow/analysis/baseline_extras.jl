# Paper Sec. 4 (baseline.tex) and Sec. 5's last paragraph: the numbers not in score_d6.jl's tables
# (results_LSTMS.md §15). Reads the reused baselines (D6_LinReg1/7/DDN, the R2 long runs).
#
#     D6_EXCLUDE_ICS=170,197,313 julia --startup-file=no --project=lib/RikFlow/analysis \
#         lib/RikFlow/analysis/baseline_extras.jl
#
# 1. bias of the ensemble mean by lead in the hindcast (policy A: the ICs all three closures keep),
#    in reference sd, per QoI;
# 2. long-run bias per 10 TU block, against the reference over the SAME block (shared forcing), in %
#    of that block's reference mean;
# 3. LinReg1's residual on the held-out 10-100 TU of the tracked record (teacher-forced): mean and sd
#    against the fitted Gaussian's;
# 4. the autocorrelation of the correction dQ in the long runs against the tracked record's, lags
#    1-200: the largest difference per QoI.
include(joinpath(@__DIR__, "score_d6.jl"))       # load_members, restrict, assemble, load_truth, LEADS
# the stdlib-only ts_* layer (the analysis project has no RikFlow): HistorySpec, build_history, scale_input
include(joinpath(SRC, "ts_history.jl"))
include(joinpath(SRC, "ts_scaling.jl"))
const OUTD = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const LABS = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
fmt6(v; d = 2) = join((@sprintf("%7.*f", d, x) for x in v), "")

truth = load_truth()
sdref = vec(std(truth.q; dims = 2))

# --- 1. hindcast bias by lead ---------------------------------------------------------------------
println("1. Hindcast bias of the ensemble mean by lead, reference sd (policy A; D6_EXCLUDE_ICS = ",
        join(sort(collect(EXCLUDE_ICS)), ","), ")")
ens = Dict(c => load_members(joinpath(OUTD, "D6_$c")) for c in ("LinReg1", "LinReg7", "DDN"))
ks = sort(intersect((e.ks for e in values(ens))...))
grid = filter(<=(minimum(e.nlead for e in values(ens))), LEADS)
@printf("   K = %d ICs, leads %s\n   %-8s %-6s %s   mean\n", length(ks), join(grid, ","), "closure", "lead", join((lpad(l, 7) for l in LABS), ""))
for c in ("LinReg1", "LinReg7", "DDN")
    fc, tr = assemble(restrict(ens[c], ks), truth, grid)
    b = dropdims(mean(dropdims(mean(fc; dims = 3); dims = 3) .- tr; dims = 1); dims = 1) ./ sdref  # nq x L
    for (j, l) in pairs(grid)
        @printf("   %-8s %-6d %s %7.2f\n", c, l, fmt6(b[:, j]), mean(b[:, j]))
    end
end

# --- 2. long-run bias per 10 TU block -----------------------------------------------------------------
println("\n2. Long-run bias per 10 TU block, % of the reference mean over the same block (mean over replicas; range over QoIs)")
function longruns(c)
    dir, pat = c == "DDN" ? (joinpath(OUTD, "TO_DDN"), r"^DDN_data_online_tsim100\.0_replica\d\.jld2$") :
                            (joinpath(OUTD, "TO_LRS", c), r"^data_online_tsim100\.0_replica\d\.jld2$")
    fs = sort(filter(f -> occursin(pat, f), readdir(dir)))
    return [jldopen(io -> (q = io["data_online"].q, dQ = io["data_online"].dQ), joinpath(dir, f)) for f in fs]
end
blocks = [((b - 1) * 4000 + 1):(b * 4000 + 1) for b in 1:10]
runs = Dict(c => longruns(c) for c in ("LinReg1", "LinReg7", "DDN"))
for c in ("LinReg1", "LinReg7", "DDN")
    rs = runs[c]
    @printf("   %-8s (%d runs)", c, length(rs))
    for bl in blocks
        refm = vec(mean(truth.q[:, bl]; dims = 2))
        m = mean(vec(mean(r.q[:, bl]; dims = 2)) ./ refm .- 1 for r in rs) .* 100
        @printf("  [%+.1f,%+.1f]", minimum(m), maximum(m))
    end
    println()
end

# --- 3. LinReg1's residual on 10-100 TU -----------------------------------------------------------------
println("\n3. LinReg1's residual, teacher-forced on the tracked record (scaled units)")
rec = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
m1 = load(joinpath(OUTD, "TO_LRS", "LinReg1", "LinReg.jld2"))
C1 = permutedims(Matrix{Float64}(m1["c"]))
s1 = m1["scaling"].in_scaling
hist = HistorySpec(; h = 5, n_qoi = 6, hist_var = :q_star_q, include_predictor = true)
X, Y, st = build_history(hist, scale_input(rec["q_star"], s1), scale_input(rec["q"], s1))
sdist = m1["stoch_distr"]       # read without Distributions here: the covariance is the PDMat's `mat`
sdfit = sqrt.(diag(Matrix{Float64}(hasproperty(sdist.Σ, :mat) ? sdist.Σ.mat : sdist.Σ)))
for (lab, lo, hi) in (("training 1-10 TU", 1.0, 10.0), ("held out 10-100 TU", 10.0, 100.0))
    rows = findall(n -> lo <= n * DT <= hi, st)
    R = Y[rows, :] .- X[rows, :] * C1
    @printf("   %-20s mean / fitted sd %s | sd / fitted sd %s\n", lab, fmt6(vec(mean(R; dims = 1)) ./ sdfit; d = 3),
            fmt6(vec(std(R; dims = 1)) ./ sdfit))
end

# --- 4. ACF of dQ in the long runs against the tracked record ---------------------------------------------
println("\n4. ACF of dQ, long runs vs tracked record (0-100 TU), lags 1-200: max |difference| per QoI (worst replica, and replica mean)")
dqref = rec["q"][:, 2:end] .- rec["q_star"]
acfr = [autocorr(collect(dqref[i, :]), 200) for i in 1:6]
for c in ("LinReg1", "LinReg7", "DDN")
    worst = zeros(6); avg = zeros(6)
    for r in runs[c]
        for i in 1:6
            a = autocorr(collect(Float64.(r.dQ[i, 101:end])), 200)
            d = maximum(abs, a[2:end] .- acfr[i][2:end])
            worst[i] = max(worst[i], d); avg[i] += d / length(runs[c])
        end
    end
    @printf("   %-8s worst %s | mean %s\n", c, fmt6(worst), fmt6(avg))
end
