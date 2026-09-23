# Every trained M4 model on ONE validation set, so their losses are comparable numbers.
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/analysis/m4_common_val.jl
#
# ⚠️ **Runs under the training project, not `--project=analysis`.** `elbo` lives in the Lux
# extension. `postrun_lstm.jl` and `ou_replay.jl` are the other two exceptions of this kind and
# both say so in their own headers rather than leaving a reader to discover it.
#
# ---------------------------------------------------------------------------------------------
# Why this exists
# ---------------------------------------------------------------------------------------------
#
# 🔴 **Each fit reports a validation loss measured on ITS OWN validation set, and those sets are
# not the same.** The fits made before 2026-09-18 split the record by position in the segment
# list, so their scored validation rows are 2901:3599; the row-based split that replaced it scores
# 2980:3599. Two numbers called "best val" that were never computed on the same rows, with nothing
# in either file saying so.
#
# That is not a small effect here. The pre-split `StochLSTM2` fit reads 8.13e-4 and the post-split
# re-fit of the SAME configuration at the same update count reads 1.71e-3 -- 2.1x apart, where
# `results_LSTMS.md` predicted the split would move "the values in the third digit". Until every
# model is scored on one set there is no way to tell how much of that gap is the fits and how much
# is the yardstick.
#
# 🔑 **The set used here is the row-split validation block, segmented at `stride = 400`** (Rik,
# 2026-09-18: *"just use 400 stride there so there is no overlap"*). At `stride = L - burn` the
# scored windows exactly tile the block, so every row is scored exactly once and no row is
# weighted twice -- which a shorter stride would do, and which would make the number depend on the
# segmentation rather than on the model.
#
# 🔑 **Scored rows 2980:3599 are disjoint from BOTH training blocks.** The pre-split fits trained
# on scored rows 101:2900 and the post-split ones on 101:2879, so this set leaks into neither and
# is a fair held-out set for every model in the table. That is the property that makes one common
# set possible at all, and it is checked below rather than asserted here.
#
# ⚠️ **The reported number is the RECONSTRUCTION term, `beta = 0`.** The models differ in `beta`,
# and the KL is weighted by it, so a full ELBO would compare fits partly on how hard each one was
# penalised. The reconstruction term is the same functional for every `emission = :none` cell --
# a sum of squares per scored step -- so it is the one column they can share. The ELBO at each
# model's own `beta` is reported beside it, and is NOT comparable across different `beta`.
#
# 🔴 **The `:lstm` control is on a different scale and gets its own section.** With
# `emission = :constant` the reconstruction term is a Gaussian log-density, not a sum of squares
# (`results_LSTMS.md` §2), so it shares no axis with the latent cells. Same trap as fig11.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Random
using Statistics
using Printf

const RF = RikFlow

TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
include(joinpath(@__DIR__, "..", "exp_square_HIT", "tools", "m4_data.jl"))

const CELL = parse(Int, get(ENV, "RIKFLOW_M4_CELL", "2"))
const VAL_FRAC = parse(Float64, get(ENV, "RIKFLOW_M4_VAL_FRAC", "0.2"))

inputs = load(joinpath(TO, "inputs_lstm.jld2"), "inputs")
cfg = inputs[CELL]
# The same resolution order as the three M4 drivers: `RIKFLOW_QOI_CACHE`, else a cache discovered
# under `analysis/data/`, else the 2.7 GB record.
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(TO, "..", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
rec = load_m4_qois(; track_file)
a, b = cfg.train_range
@info "common validation set" cell = cfg.name train_range = cfg.train_range val_frac = VAL_FRAC

# --- the window and the scaling, exactly as `11_train_StochLSTM.jl` builds them -----------------
_, in_scaling = RF._normalise(rec.q[:, a:(b - 1)]; normalization = cfg.normalization)
qs = RF.scale_input(rec.q[:, a:b], in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b - 1)], in_scaling)
hist = RF.HistorySpec(; h = cfg.h, n_qoi = size(rec.q, 1), cfg.hist_var, cfg.include_predictor)
X, Y, steps = RF.build_history(hist, qss, qs)
Xc, Yc = permutedims(Float32.(X)), permutedims(Float32.(Y))

ncol = length(steps)
ntrain = floor(Int, (1 - VAL_FRAC) * ncol)
L, burn = cfg.L, cfg.burn

# 🔑 `stride = L - burn` is the default and is passed explicitly, because it is the whole point of
# this set rather than an incidental default.
val_raw = RF.segment_indices(view(steps, (ntrain + 1):ncol); L, burn, stride = L - burn)
val_segs = [(; rows = s.rows .+ ntrain, score = s.score .+ ntrain) for s in val_raw]
scored = sort(reduce(vcat, (collect(s.score) for s in val_segs)))

# 🔴 The two properties the set has to have, checked rather than trusted.
allunique(scored) ||
    error("scored validation rows repeat -- the stride is overlapping this set, which re-weights " *
          "rows and makes the loss a function of the segmentation")
first(scored) > 2900 ||
    error("scored validation starts at $(first(scored)), which the PRE-split fits trained on " *
          "(they scored up to 2900). This set is not held out for them.")
@info "validation block" rows = (ntrain + 1, ncol) segments = length(val_segs) scored_rows = length(scored) span = (first(scored), last(scored))

# --- the fixed batches, and one fixed noise draw per latent dimension ---------------------------
function group_by_length(segs)
    d = Dict{Int,Vector{Int}}()
    for (i, s) in enumerate(segs)
        push!(get!(d, length(s.rows), Int[]), i)
    end
    return d
end

function stack(segs, idx)
    Ls = length(segs[first(idx)].rows)
    Xb = Array{Float32,3}(undef, size(Xc, 1), Ls, length(idx))
    Yb = Array{Float32,3}(undef, size(Yc, 1), Ls, length(idx))
    for (k, i) in enumerate(idx)
        Xb[:, :, k] = view(Xc, :, segs[i].rows)
        Yb[:, :, k] = view(Yc, :, segs[i].rows)
    end
    s1 = segs[first(idx)]
    sc = (first(s1.score) - first(s1.rows) + 1):(last(s1.score) - first(s1.rows) + 1)
    return Xb, Yb, sc, length(sc) * length(idx)
end

groups = sort(collect(group_by_length(val_segs)); by = first)
batches = [stack(val_segs, idx) for (_, idx) in groups]

# 🔑 **One draw per `n_latent`, from a fixed seed, reused for every model of that width.** The
# models are not all the same width, so a single array cannot serve them all; what can be held
# fixed is that any two models with the same `n_latent` see the SAME noise. Redrawing per model
# would mix "this model is worse" with "this model drew worse", which is the thing the fixed
# validation epsilons inside `train_stochlstm` exist to prevent, one level up.
const EPS_CACHE = Dict{Int,Vector{Array{Float32,3}}}()
function eps_for(nz)
    get!(EPS_CACHE, nz) do
        rng = Xoshiro(20_260_918)
        [randn(rng, Float32, nz, size(Xb, 2), size(Xb, 3)) for (Xb, _, _, _) in batches]
    end
end

"The reconstruction term (`beta = 0`) and the ELBO at `beta`, both per scored step."
function score_model(spec, ps, beta)
    eb = eps_for(spec.n_latent)
    num0, numb, den = 0.0, 0.0, 0
    for (k, (Xb, Yb, sc, ns)) in enumerate(batches)
        num0 += RF.elbo(spec, ps, Xb, Yb, sc, eb[k]; beta = 0.0) * ns
        numb += RF.elbo(spec, ps, Xb, Yb, sc, eb[k]; beta) * ns
        den += ns
    end
    return (; recon = num0 / den, elbo = numb / den)
end

# --- collect every trained model ----------------------------------------------------------------
models = NamedTuple[]

for f in sort(filter(x -> startswith(x, "stride_scan_") && endswith(x, ".jld2"), readdir(TO)))
    d = load(joinpath(TO, f))
    spec = d["spec"]
    beta = d["cfg"].beta
    tag = replace(f, "stride_scan_" => "", ".jld2" => "")
    for r in d["results"]
        push!(models, (; label = "stride $(r.stride) b$(r.batch)", source = tag, arch = spec.arch,
                       emission = spec.emission, beta, spec, ps = r.ps,
                       own_val = r.best_val))
    end
end

for dir in sort(filter(x -> startswith(x, "StochLSTM") && isdir(joinpath(TO, x)), readdir(TO)))
    for f in sort(filter(x -> endswith(x, ".jld2"), readdir(joinpath(TO, dir))))
        d = load(joinpath(TO, dir, f))
        e = d["extras"]
        haskey(e, :ps) || (@warn "no `ps` stored, so no ELBO: $dir/$f"; continue)
        fit = RF.load_stochlstm(joinpath(TO, dir, f))
        push!(models, (; label = "$dir $(splitext(f)[1])", source = "cell fit",
                       arch = fit.spec.arch, emission = fit.spec.emission, beta = e.cfg.beta,
                       spec = fit.spec, ps = e.ps, own_val = e.losses.best_val))
    end
end

isempty(models) && error("no trained models found under $TO")
@info "models" n = length(models)

rows = NamedTuple[]
for m in models
    s = score_model(m.spec, m.ps, m.beta)
    push!(rows, (; m.label, m.source, m.arch, m.emission, m.beta, m.own_val, s.recon, s.elbo))
end

# --- report ---------------------------------------------------------------------------------------
# 🔴 `emission = :none` and `:constant` are printed as two tables, not two rows of one, because a
# sum of squares and a Gaussian log-density in the same column invite exactly the comparison that
# is meaningless.
function table(title, rs)
    isempty(rs) && return
    println("\n", title)
    @printf("%-28s %-12s %-8s %-8s %12s %12s %12s %8s\n", "model", "source", "arch", "beta",
            "recon (b=0)", "elbo (own b)", "own val", "ratio")
    println("-"^108)
    ref = minimum(r.recon for r in rs)
    for r in sort(rs; by = x -> x.recon)
        @printf("%-28s %-12s %-8s %-8g %12.5g %12.5g %12.5g %8.2f\n",
                r.label, r.source, r.arch, r.beta, r.recon, r.elbo, r.own_val, r.recon / ref)
    end
end

table("emission = :none  --  reconstruction is a sum of squares per scored step",
      [r for r in rows if r.emission === :none])
table("emission = :constant  --  DIFFERENT UNITS (Gaussian log-density), not comparable above",
      [r for r in rows if r.emission !== :none])

println("\n🔑 `own val` is each fit's own reported best validation loss, on ITS OWN validation set.")
println("   Where it disagrees with `recon`, the disagreement IS the point: those numbers were")
println("   never measured on the same rows. `ratio` is against the best model in its own table.")

mkpath(joinpath(@__DIR__, "output"))
out = joinpath(@__DIR__, "output", "m4_common_val.jld2")
jldsave(out; rows, scored_span = (first(scored), last(scored)), scored_rows = length(scored),
        n_segments = length(val_segs), stride = L - burn, L, burn, val_frac = VAL_FRAC,
        train_range = cfg.train_range, cell = cfg.name)
println("\nwrote $out")
