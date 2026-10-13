# Do the two implementations draw common random numbers? (Rik, 2026-10-13: option (b) for the
# control run, paper Sec. 6.1.)
#
# LinReg1 (first implementation, `LinReg`) and its exact translation into the second (`StochLSTM`,
# `TO_LSTM/diag/control_LinReg1`, results_LSTMS.md Sec. 14) are built as the D6 driver builds them:
# the same member seed `Xoshiro(seed)`, the same 100-step warm-up replayed from the record. Both are
# then fed the SAME recorded q* sequence for L steps (teacher-forced, so the LES plays no part) and
# their sampled corrections are compared step by step, in sd(dQ) units:
#
#   same seed      LinReg1(seed)  vs control(seed)       common random numbers -> Float32 round-off
#   other seed     LinReg1(seed)  vs LinReg1(seed + 1)    independent draws     -> O(residual sd)
#   mean only      LinReg1 mean   vs control mean         (stochastic = false), the GATE 2 number
#
# Usage (from lib/RikFlow): julia --project=. analysis/crn_check.jl

using RikFlow, JLD2, Random, Statistics, Printf
const RF = RikFlow
const HERE = @__DIR__
const OUT = joinpath(HERE, "..", "exp_square_HIT", "output")
const REC = load(joinpath(HERE, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const QS = Float64.(REC["q_star"])
const DQ = Float64.(REC["q"][:, 2:end]) .- QS            # dQ^n = q^{n+1} - q^{n*}, as plot_paper_secs45.jl
const NQ = size(QS, 1)
const NWARM = 100
const L = 60      # teacher-forced on the recorded q*, the closed history has no restoring force and runs away after ~200 steps (results_LSTMS.md Sec. 7); 60 steps stay well inside
const LRS = joinpath(OUT, "TO_LRS", "LinReg1", "LinReg.jld2")
const FIT = RF.load_stochlstm(joinpath(OUT, "TO_LSTM", "diag", "control_LinReg1", "StochLSTM_seed1.jld2"))
const LABELS = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const SD = vec(std(DQ[:, 400:4000]; dims = 2))

function linreg(seed, s)
    h = load(LRS, "hist_len"); hv = load(LRS, "hist_var")
    qh = zeros(NQ, h); hv == :q_star_q && (qh = vcat(qh, qh))
    return RF.LinReg(LRS, Xoshiro(seed), Array; q_hist = qh, spinnup_data = DQ[:, s:(s + NWARM - 1)])
end
control(seed, s; stochastic = true) =
    RF.StochLSTM(FIT.spec, FIT.weights, FIT.scaling; spinnup_data = DQ[:, s:(s + NWARM - 1)],
                 rng = Xoshiro(seed), stochastic)

"The L sampled corrections after the warm-up, raw units, NQ x L."
function run(m, s)
    for n in s:(s + NWARM - 1)
        RF.get_next_item_timeseries(m, QS[:, n])
    end
    out = zeros(NQ, L)
    for (j, n) in enumerate((s + NWARM):(s + NWARM + L - 1))
        out[:, j] .= Array(RF.get_next_item_timeseries(m, QS[:, n]))
    end
    return out
end

starts = round.(Int, range(4400, 39000; length = 12))      # held-out window, 11-97.5 TU
seeds = [hash((:d6, k, 1)) for k in 1:length(starts)]

# --- A. the first sampled step, through the full prediction path (the draw is the first one) ------
println("A. first sampled step after the warm-up, |difference| in sd(dQ), max over the 6 QoIs")
same1 = Float64[]; other1 = Float64[]
for (s, seed) in zip(starts, seeds)
    a = run(linreg(seed, s), s)[:, 1]; b = run(control(seed, s), s)[:, 1]; c = run(linreg(seed + 1, s), s)[:, 1]
    push!(same1, maximum(abs.(a .- b) ./ SD)); push!(other1, maximum(abs.(a .- c) ./ SD))
end
@printf("   same seed (LinReg1 vs control):     %.1e to %.1e
", extrema(same1)...)
@printf("   other seed (LinReg1 vs LinReg1):    %.1e to %.1e
", extrema(other1)...)

# --- B. the noise sequences themselves, N steps, no dynamics --------------------------------------
# LinReg: `draw_eta` per step (scaled level units) -> physical through its out_scaling sigma.
# control: `sample_emission!` per step with the mean set to zero (scaled dQ units) -> physical through
# its out_scaling sigma. Each is called exactly as the closure calls it, one call per step, so any
# difference in how many deviates a step consumes shows up as a desynchronized sequence.
const N = 2000
lrs_sig = vec(Array(load(LRS, "scaling").out_scaling.sigma))
ctl_sig = vec(Float64.(FIT.scaling.out_scaling.sigma))
println("
B. noise sequences over $N steps, physical units / sd(dQ)")
for seed in seeds[1:4]
    m1 = linreg(seed, starts[1]); m2 = control(seed, starts[1])
    e1 = zeros(NQ, N); e2 = zeros(NQ, N); e3 = zeros(NQ, N)
    for n in 1:N
        e1[:, n] .= Array(RF.draw_eta(m1)) .* lrs_sig
        fill!(m2.state.y, 0); m2.state.logd .= m2.weights.bd
        RF.sample_emission!(m2.scratch, m2.state, m2.weights, m2.spec, m2.rng)
        e2[:, n] .= Float64.(m2.scratch) .* ctl_sig
    end
    m3 = linreg(seed + 1, starts[1])
    for n in 1:N
        e3[:, n] .= Array(RF.draw_eta(m3)) .* lrs_sig
    end
    d = abs.(e1 .- e2) ./ SD; r = abs.(e1 .- e3) ./ SD; nz = std(e1; dims = 2) ./ SD
    @printf("   seed %20d: same seed max %.1e (step %d of %d), other seed median %.2f; noise sd %s sd(dQ)
",
            seed, maximum(d), argmax(vec(maximum(d; dims = 1))), N, median(r),
            join((@sprintf("%.2f", v) for v in nz), " "))
end
