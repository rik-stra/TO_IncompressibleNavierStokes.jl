# Per-QoI λ for the LRS: splice the coefficient ROWS of fitted `TO_LRS/LinReg<n>` models into one model
# (results_LSTMS §13b′, 2026-09-28).
#
#     julia --startup-file=no --project=training exp_square_HIT/tools/lrs_splice.jl <dst> <src_1> ... <src_6>
#
# e.g. `LinReg1e7 LinReg1 LinReg7 LinReg1 LinReg1 LinReg1 LinReg1` = LinReg1 with LinReg7's E[0,6] row.
#
# 🔑 Why this is exactly a ridge fit with a per-row λ, not an approximation. `5_train_LinReg.jl` solves
# `[X; sqrt(λ)P] \ [Y; 0]` with one right-hand-side column per QoI and the SAME design X and penalty P
# for all of them, so each output column is its own least-squares problem: row i of `c` depends on
# λ and on QoI i's target only. The input/output scaling is the data's (`_normalise` on the train
# range), not λ's, so every λ cell carries the same `scaling`, asserted below. What does NOT splice
# is the noise: the residual mean and covariance couple the rows, so `stoch_distr` is REFITTED here
# on the spliced residual, exactly as `fit_model` does (`fit(MvNormal, ·)`, the MLE: mean and
# uncorrected covariance), on the same training rows, rebuilt with `create_history`'s construction.
# A no-op splice (all six rows from one source) must reproduce that source's stored `stoch_distr`
# and is checked on every run.
#
# Writes `output/TO_LRS/<dst>/LinReg.jld2` (every key of `src_1`'s file, `c` and `stoch_distr`
# replaced, plus `splice_sources` and `splice_lambda`) and `parameters.jld2` (src_1's with `name`
# changed, `lambda = NaN` and `lambda_per_qoi` added -- a spliced model has no single λ). Refuses an
# existing target. An AR residual goes on top with `lrs_ar_variant.jl <dst> 2`.
# 🔒 Reads the 1-10 TU training rows only.

using RikFlow
using JLD2, LinearAlgebra, Statistics, Printf, Dates

const LRS = normpath(joinpath(@__DIR__, "..", "output", "TO_LRS"))
const REF = normpath(joinpath(@__DIR__, "..", "..", "analysis", "data",
                              "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const NQ = 6
const LABELS = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")

"""
The cell's settings from `inputs_example.jld2` (by name): older `parameters.jld2` files (before
2026-09-15) carry neither `lambda` nor `train_range`, and the inputs file is what `5_train_LinReg.jl`
was run from.
"""
function cell_params(name)
    inputs = load(joinpath(LRS, "inputs_example.jld2"), "inputs")
    i = findfirst(x -> x.name == name, inputs)
    i === nothing && error("$name is not in inputs_example.jld2")
    return inputs[i]
end

"`5_train_LinReg.jl`'s training rows (inputs with the bias column, and targets), scaled, for `m`."
function train_rows(m, p)
    @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
    tr = p.train_range
    @assert tr == (400, 4000) "train_range $(tr): this tool reads 1-10 TU only"
    q, qs = load(REF, "q", "q_star")
    sc = m["scaling"]
    q_s = RikFlow.scale_input(q[:, tr[1]:(tr[2] - 1)], sc.in_scaling)
    qs_s = RikFlow.scale_input(qs[:, tr[1]:(tr[2] - 1)], sc.in_scaling)
    dq_s = RikFlow.scale_input(q[:, (tr[1] + 1):tr[2]], sc.in_scaling)
    # create_history(h, q_star[:,2:end], [q[:,2:end]; q_star[:,1:end-1]], dQ[:,2:end]) for :q_star_q
    h = m["hist_len"]
    P = qs_s[:, 2:end]
    Q = vcat(q_s[:, 2:end], qs_s[:, 1:(end - 1)])
    T = dq_s[:, 2:end]
    X = vcat(P[:, h:end], (Q[:, (h - i + 1):(end - i + 1)] for i in 1:h)...)
    Y = T[:, h:end]
    return vcat(X, ones(1, size(X, 2))), Y
end

"MLE Gaussian of the residual, as `fit(MvNormal, R)` (R: NQ x N)."
mle(R) = (vec(mean(R; dims = 2)), Matrix(Symmetric(cov(R; dims = 2, corrected = false))))

function splice(dst, srcs; io = stdout)
    length(srcs) == NQ || error("need $NQ sources, one per QoI; got $(length(srcs))")
    ddir = joinpath(LRS, dst)
    isdir(ddir) && error("$ddir exists; refusing to overwrite")
    ms = Dict(s => load(joinpath(LRS, s, "LinReg.jld2")) for s in unique(srcs))
    ps = Dict(s => cell_params(s) for s in unique(srcs))
    m1, p1 = ms[srcs[1]], ps[srcs[1]]
    for s in unique(srcs)
        m, p = ms[s], ps[s]
        @assert m["hist_len"] == m1["hist_len"] && m["hist_var"] == m1["hist_var"] "$s: different design"
        @assert p.train_range == p1.train_range && p.normalization == p1.normalization "$s: different data"
        for k in (:in_scaling, :out_scaling), f in (:mu, :sigma)
            @assert getfield(getfield(m["scaling"], k), f) == getfield(getfield(m1["scaling"], k), f) "$s: scaling differs"
        end
    end
    X, Y = train_rows(m1, p1)
    # the no-op check, per source: stored stoch_distr == MLE of its own residual on these rows
    for s in unique(srcs)
        mu, S = mle(Y .- Matrix(ms[s]["c"]) * X)
        sd = ms[s]["stoch_distr"]
        e1 = maximum(abs.(mu .- mean(sd))) / maximum(sqrt.(diag(S)))
        e2 = maximum(abs.(S .- cov(sd))) / maximum(abs.(S))
        @printf(io, "  no-op check %-9s: |Δμ|/sd %.1e, |ΔΣ|/|Σ| %.1e\n", s, e1, e2)
        (e1 < 1e-8 && e2 < 1e-8) || error("$s: the rebuilt training rows do not reproduce its stoch_distr")
    end
    c = vcat((Matrix(ms[s]["c"])[i:i, :] for (i, s) in enumerate(srcs))...)
    mu, S = mle(Y .- c * X)
    sd1 = m1["stoch_distr"]
    Dmod = parentmodule(typeof(sd1))
    stoch = Dmod.MvNormal(mu, S)
    lam = [ps[s].lambda for s in srcs]
    println(io, "==== $dst: rows ", join(("$(LABELS[i]) <- $(srcs[i]) (λ = $(lam[i]))" for i in 1:NQ), ", "))
    @printf(io, "  residual sd (scaled) per QoI: %s\n", join((@sprintf("%.3e", sqrt(S[i, i])) for i in 1:NQ), " "))
    mkpath(ddir)
    d = Dict{String,Any}(m1)
    d["c"] = c
    d["stoch_distr"] = stoch
    d["splice_sources"] = collect(srcs)
    d["splice_lambda"] = lam
    d["splice_provenance"] = (; built = string(now()), script = "exp_square_HIT/tools/lrs_splice.jl",
                              noise = "MLE Gaussian refitted on the spliced residual, rows 400-4000")
    save(joinpath(ddir, "LinReg.jld2"), d)
    save(joinpath(ddir, "parameters.jld2"), "parameters",
         merge(load(joinpath(LRS, srcs[1], "parameters.jld2"), "parameters"),
               (; name = dst, lambda = NaN, lambda_per_qoi = Tuple(lam), train_range = p1.train_range)))
    # round trip
    m2 = load(joinpath(ddir, "LinReg.jld2"))
    @assert Matrix(m2["c"]) == c && cov(m2["stoch_distr"]) ≈ S
    println(io, "  wrote $ddir")
    return (; c, mu, S)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 + NQ || error("usage: lrs_splice.jl <dst> <src_1> ... <src_6>")
    splice(ARGS[1], ARGS[2:end])
end
