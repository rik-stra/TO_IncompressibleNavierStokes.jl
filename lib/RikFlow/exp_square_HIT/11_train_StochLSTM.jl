# Fit one M4 configuration.
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/11_train_StochLSTM.jl <model_index> [seed]
#
# ⚠️ Runs under `--project=lib/RikFlow/training`, not `--project=lib/RikFlow`: training needs the
# Lux extension, which is triggered by Lux + Optimisers + Zygote together. The *deployed* model
# needs none of them -- see `src/ts_lstm.jl`'s header.
#
# 🔑 S6 asks for **5 training seeds per neural cell, spread reported, median-seed fit online.**
# The optional second argument fits one seed, so a SLURM array can put them on separate tasks; with
# no argument all `n_seeds` are fitted in sequence. `12_online_StochLSTM.jl` picks the median.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Random
using Statistics
using Printf

const RF = RikFlow

length(ARGS) >= 1 || error("usage: julia 11_train_StochLSTM.jl <model_index> [seed]")
model_index = parse(Int, ARGS[1])
seed_arg = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing

Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

TO_folder = @__DIR__() * "/output/TO_LSTM"

# The tracking record. 🔴 The `_f64_lmwray3` suffix is load-bearing: a Float64/LMWray3 record must
# never be confusable with the archived Float32/RK44 one, which is otherwise identically named and
# sits on the pre-`09954be1` Nyquist convention -- a different dynamical system, not a less
# accurate measurement of this one (claude_memory.md #45, #46).
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 @__DIR__() * "/output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2")
# The extracted QoI cache is an equally good source and is 7 MB rather than 2.7 GB, so it is
# preferred when present. `analysis/extract_qois.jl` writes it.
qoi_cache = get(ENV, "RIKFLOW_QOI_CACHE", "")

function load_qois()
    if !isempty(qoi_cache)
        isfile(qoi_cache) || error("RIKFLOW_QOI_CACHE=$qoi_cache does not exist")
        d = load(qoi_cache)
        @info "reading QoIs from the cache" qoi_cache
        return (; q = d["q"], q_star = d["q_star"])
    end
    isfile(track_file) || error("neither RIKFLOW_QOI_CACHE nor the tracking record is present:\n" *
                                "  track_file = $track_file")
    @info "reading QoIs from the tracking record (slow -- consider the cache)" track_file
    d = load(track_file, "data_track")
    return (; d.q, d.q_star)
end

inputs = load(TO_folder * "/inputs_lstm.jld2", "inputs")
1 <= model_index <= length(inputs) ||
    error("model_index $model_index is out of range 1:$(length(inputs))")
cfg = inputs[model_index]
@info "fitting $(cfg.name)" arch=cfg.arch h=cfg.h beta=cfg.beta

rec = load_qois()
a, b = cfg.train_range
size(rec.q, 2) >= b || error("the record has $(size(rec.q, 2)) columns but train_range needs $b")

# --- standardisation, exactly as `5_train_LinReg.jl` does it -----------------------------------
# One `Scaling` for input and output: the target is the level `q^{n+1}` and the input carries
# `q^{n*}` and the lagged `q`, so all of it is q-space and one set of statistics is right.
_, in_scaling = RF._normalise(rec.q[:, a:(b - 1)]; normalization = cfg.normalization)
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)

# --- the regressor, from the SHARED builder ----------------------------------------------------
# `build_history` wants `q` one column longer than `q_star`, because column 1 of `q` is t = 0.
# V1 and V2 cover this alignment; M4 introduces no new layout.
hist = RF.HistorySpec(; h = cfg.h, n_qoi = size(rec.q, 1), cfg.hist_var, cfg.include_predictor)
qs = RF.scale_input(rec.q[:, a:b], in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b - 1)], in_scaling)
X, Y, steps = RF.build_history(hist, qss, qs)
@info "regressor built" rows=size(X, 1) features=size(X, 2) span=(first(steps), last(steps))

# `emission` is read with a fallback so a table written before the field existed still loads.
# 🔴 The fallback is `:none`, not `:state_dependent` (Rik, 2026-09-18): the Gaussian emission head
# is off, so a table that does not name an emission gets the source's deterministic decoder rather
# than a second noise channel nobody asked for.
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, cfg.arch, cfg.uclip,
                   emission = get(cfg, :emission, :none))

# `build_history` is row-major (one row per step) because the linear cells solve a least-squares
# system with it; the recurrence wants time last.
Xc, Yc = permutedims(X), permutedims(Y)

# A smoke-test override. `RIKFLOW_M4_EPOCHS=5` makes the whole driver runnable in a minute, which
# is what the pre-flight needs; it is not a way to run the experiment.
epochs = parse(Int, get(ENV, "RIKFLOW_M4_EPOCHS", string(cfg.epochs)))
epochs == cfg.epochs || @warn "epochs overridden by RIKFLOW_M4_EPOCHS" epochs config=cfg.epochs


# 🔴 Device placement. `M4_DEVICE=cpu` (the default) or `cuda`. `m4_device` refuses `cuda` when no
# device is functional rather than falling back to the host, because a silent fallback would report
# a GPU run that took CPU time. ⚠️ **The GPU path has never been run on a GPU** -- it is verified
# only against `JLArrays`, which enforces the same no-scalar-indexing semantics on the host (V54).
# And expect it to be SLOWER at the current geometry: `results_LSTMS.md` §5.
const DEVICE = RF.m4_device(get(ENV, "M4_DEVICE", "cpu"))
@info "device" M4_DEVICE=get(ENV, "M4_DEVICE", "cpu")

seeds = seed_arg === nothing ? (1:cfg.n_seeds) : (seed_arg:seed_arg)
out_dir = TO_folder * "/$(cfg.name)"
mkpath(out_dir)

summaries = NamedTuple[]
for s in seeds
    @info "seed $s of $(cfg.n_seeds)"
    # `stride` is read with a fallback so a table written before it existed still loads and keeps
    # the behaviour it had: `L - burn`, at which the scored windows exactly tile the record.
    ps, hist_loss = RF.train_stochlstm(spec, Xc, Yc, steps;
                                       cfg.L, cfg.burn, epochs, cfg.batch, cfg.lr,
                                       cfg.beta, cfg.val_frac, seed = s, device = DEVICE,
                                       stride = get(cfg, :stride, cfg.L - cfg.burn))
    w = RF.LSTMWeights(ps, spec)
    path = "$(out_dir)/StochLSTM_seed$(s).jld2"
    # 🔑 `ps` is stored alongside `w`. They are the same model in two parametrisations -- `w`
    # carries the correlation matrix the solver uses, `ps` the free precision factor training used
    # -- and the IWAE bound is defined on `ps`, so without it `analysis/postrun_lstm.jl` cannot
    # report a likelihood at all. The model is tiny; the duplication costs kilobytes.
    RF.save_stochlstm(path, spec, w, scaling;
                      cfg, seed = s, train_range = cfg.train_range, track_file, qoi_cache,
                      losses = hist_loss, steps_span = (first(steps), last(steps)), ps)
    # 🔴 `best_val`, not `val[end]`. `train_stochlstm` returns the best-validation iterate, so the
    # last epoch's loss is not the loss of the model that was saved -- and on R1's record the two
    # differed by ~8 nats, enough to invert the architecture ranking. `final_val` is kept beside it
    # precisely so a run that ended far from its best is visible rather than silently averaged in.
    push!(summaries, (; seed = s, best_val = hist_loss.best_val, best_epoch = hist_loss.best_epoch,
                      final_train = hist_loss.train[end], final_val = hist_loss.val[end], path))
    @printf("  seed %d: best val %.4f (epoch %d)  final val %.4f  -> %s\n",
            s, hist_loss.best_val, hist_loss.best_epoch, hist_loss.val[end], basename(path))
    if hist_loss.best_epoch < 0.8 * epochs
        @warn "seed $s peaked early and then got worse -- check the lr schedule" best_epoch =
            hist_loss.best_epoch epochs
    end
end

# The seed spread S6 asks to be reported, and the median seed the online driver deploys.
if length(summaries) > 1
    vals = [s.best_val for s in summaries]
    med = summaries[sortperm(vals)[cld(length(vals), 2)]]
    jldsave("$(out_dir)/seed_summary.jld2"; summaries, median_seed = med.seed,
            val_mean = mean(vals), val_std = std(vals))
    @printf("\nval across %d seeds: mean %.4f  std %.4f  median seed %d\n",
            length(vals), mean(vals), std(vals), med.seed)
end
