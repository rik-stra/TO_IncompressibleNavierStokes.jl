# Do overlapping (shorter-stride) training segments buy anything?
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_stride_scan.jl [cell] [seed]
#
# Environment: `RIKFLOW_QOI_CACHE` (as the training driver), `RIKFLOW_M4_EPOCHS` (the epoch CAP
# per point, default 3000) and `RIKFLOW_M4_STOP_PATIENCE` (epochs past the best before stopping,
# default 100).
#
# ---------------------------------------------------------------------------------------------
# The question, and why it needs a matched budget
# ---------------------------------------------------------------------------------------------
#
# At the default `stride = L - burn` the scored windows exactly TILE the record: every row is
# scored once per epoch, and at `L = 500` the 2879-row training block leaves **7 segments** --
# far fewer than `batch`, so the batch is unreachable and an epoch is **2 updates** (the 7 fall
# into two LENGTH groups, 6 x 500 and 1 x 479, and each group is batched separately). Shortening
# the stride overlaps the segments and makes more of them: 25, 49, ~140 at stride 100, 50, 20.
#
# 🔴 **This is augmentation, not data.** The same rows are re-scored at different offsets inside a
# segment, so the extra gradients are correlated and `k x` the segments is not `k x` the
# information. Two things it can genuinely buy, and they are different:
#
#   (a) **more optimiser steps per epoch**, which is a pure budget effect and has nothing to do
#       with the overlap -- lowering `batch` at the tiling stride buys the same thing;
#   (b) **variation in how long the hidden state has been charged** when a given row is scored,
#       which is a real regulariser for a recurrence and is what only the overlap can give.
#
# 🔴 **Every point now runs to CONVERGENCE, not to a shared budget** (Rik, 2026-09-18).
# Matching updates was the earlier policy and it failed in a way worth recording: the plateau
# scheduler counts EPOCHS, so fixing the update count un-fixed the epoch count, and the three
# points with the fewest epochs got two learning-rate decays instead of six, ended two decades
# above `min_lr`, and had their best iterate at the final epoch. Their numbers were where the
# budget stopped, not where the fit went. Matching epochs instead would have been no better --
# then the shortest stride simply takes five times the updates, which is (a) wearing (b)'s
# clothes.
#
# 🔑 **A converged fit is where its objective took it, so the budget disparity stops
# mattering.** `stop_patience` ends each point once the lr is at `min_lr` and nothing has improved
# for 100 epochs; `RIKFLOW_M4_EPOCHS` is only the cap that bounds the job. Quality is then the
# converged validation loss and cost is the updates, row-steps and wall time it took -- reported
# in their own columns rather than forced equal. The `(400, batch = 2)` row still separates (a)
# from (b): it is the one point with more steps and no overlap.
#
# ⚠️ **Matched updates is NOT matched compute at `batch = 32`.** A wider batch means one update
# covers more segments, so at `stride = 50` an update sees 32 segments where the tiling stride can
# only ever offer 7 -- about 9x the row-steps for the same update count. That is the intended
# asymmetry (an update is an update, and a bigger one is the point of a bigger batch), but it does
# mean the wall times in the table are not comparable and must not be read as a cost ranking.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Random
using Statistics
using Printf

const RF = RikFlow

cell_index = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1

Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(TO_folder, "..",
                          "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
const EPOCHS = parse(Int, get(ENV, "RIKFLOW_M4_EPOCHS", "3000"))
const STOP_PATIENCE = parse(Int, get(ENV, "RIKFLOW_M4_STOP_PATIENCE", "100"))
# 🔴 Optimiser updates between validations. Everything paced by it -- the plateau rule and the
# early stop -- is then paced in UPDATES, identically at every point. Paced by the epoch instead,
# as it was until 2026-09-18, they run at 1 update per validation at the tiling stride and 3 at
# `stride = 20`, which is the confound the whole scan exists to avoid.
const VAL_EVERY = parse(Int, get(ENV, "RIKFLOW_M4_VAL_EVERY", "2"))

# 🔑 One QoI resolver for all three M4 drivers -- each used to carry its own.
include(joinpath(@__DIR__, "m4_data.jl"))

inputs = load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")
cfg = inputs[cell_index]
rec = load_m4_qois(; track_file)
a, b = cfg.train_range

# --- exactly the driver's data preparation ------------------------------------------------------
_, in_scaling = RF._normalise(rec.q[:, a:(b - 1)]; normalization = cfg.normalization)
hist = RF.HistorySpec(; h = cfg.h, n_qoi = size(rec.q, 1), cfg.hist_var, cfg.include_predictor)
qs = RF.scale_input(rec.q[:, a:b], in_scaling)
qss = RF.scale_input(rec.q_star[:, a:(b - 1)], in_scaling)
X, Y, steps = RF.build_history(hist, qss, qs)
spec = RF.LSTMSpec(; hist, cfg.n_hidden, cfg.n_latent, cfg.n_encoder, cfg.arch, cfg.uclip,
                   emission = get(cfg, :emission, :none))
Xc, Yc = permutedims(X), permutedims(Y)

# The training block is what `train_stochlstm` segments, so the segment count has to be counted on
# the same rows -- not on the whole record.
ncol = length(steps)
ntrain = floor(Int, (1 - cfg.val_frac) * ncol)

# 🔑 `m4_geometry` is shared with the smoke's walltime estimate, so the two cannot disagree.
geometry(stride, batch) = ((g = m4_geometry(steps, ntrain; cfg.L, cfg.burn, stride, batch));
                           (g.nseg, g.upd, g.scored))

# (stride, batch). 🔑 The second row is the CONTROL: the tiling stride with a batch small enough to
# give a comparable number of updates. If the shorter strides beat the tiling stride only as much
# as this row does, the gain was (a) -- more steps -- and the overlap itself bought nothing.
# 🔴 Strides 100 / 50 / 20 (Rik, 2026-09-18), replacing 200/100/50. Row 1 is the current default
# and is the reference every other row is read against -- without it "overlap helps" has nothing to
# be true of. Row 2 is the control described above; it is the only row that separates "more
# optimiser steps" from "overlapping windows", and it costs one point. ⚠️ Both are additions to the
# three values asked for: say so rather than let them look like part of the request.
POINTS = [(cfg.L - cfg.burn, cfg.batch),     # 400: the current default, the baseline
          (cfg.L - cfg.burn, 2),             # control: more steps, no overlap
          (100, cfg.batch),
          (50, cfg.batch),
          (20, cfg.batch)]

@info "M4 stride scan" cell=cfg.name arch=cfg.arch beta=cfg.beta seed epoch_cap=EPOCHS stop_patience=STOP_PATIENCE val_every=VAL_EVERY L=cfg.L burn=cfg.burn
@info "data" rows=ncol train_rows=ntrain


# 🔴 Device placement. `M4_DEVICE=cpu` (the default) or `cuda`. `m4_device` refuses `cuda` when no
# device is functional rather than falling back to the host, because a silent fallback would report
# a GPU run that took CPU time. ⚠️ **The GPU path has never been run on a GPU** -- it is verified
# only against `JLArrays`, which enforces the same no-scalar-indexing semantics on the host (V54).
# And expect it to be SLOWER at the current geometry: `results_LSTMS.md` §5.
const DEVICE = RF.m4_device(get(ENV, "M4_DEVICE", "cpu"))
@info "device" M4_DEVICE=get(ENV, "M4_DEVICE", "cpu")

# 🔴 Probe before committing. A point is 3000 updates with no output until it finishes;
# on an untried device that is an unbounded silence. This prints the per-epoch cost first.
let (s1, b1) = POINTS[1]
    # ⚠️ The projection is against the CAP, so it is an upper bound: early stopping can only
    # make a point cheaper than this says.
    m4_probe_step(spec, Xc, Yc, steps; device = DEVICE, stride = s1, batch = b1, cfg,
                  epochs_planned = EPOCHS,
                  label = "device probe ($(get(ENV, "M4_DEVICE", "cpu")), stride $s1)")
end

# 🔴 The output path is fixed BEFORE the loop, because every point writes to it. A scan killed at
# its walltime then keeps everything that finished -- see `m4_save_progress`.
out = joinpath(TO_folder, "stride_scan_$(cfg.name)_seed$(seed).jld2")

results = NamedTuple[]
for (stride, batch) in POINTS
    nseg, upd_per_epoch, scored = geometry(stride, batch)
    m4_phase("point: stride $stride batch $batch -- $nseg segments, $upd_per_epoch updates/epoch, cap $EPOCHS epochs")
    t0 = time()
    # 🔑 `ps` is KEPT, not discarded. It is ~2 950 Float32 -- 12 kB -- and without it a point that
    # took twenty minutes leaves only a loss curve, so the winning configuration would have to be
    # refitted before it could be deployed or inspected. `train_stochlstm` returns it on the host.
    ps, h = RF.train_stochlstm(spec, Xc, Yc, steps;
                              cfg.L, cfg.burn, stride, epochs = EPOCHS, batch, cfg.lr,
                              cfg.beta, cfg.val_frac, seed, verbose = true, device = DEVICE,
                              stop_patience = STOP_PATIENCE, val_every = VAL_EVERY)
    # 🔑 The geometry is taken from the HISTORY, not from `geometry(...)`. The helper predicts
    # it for the walltime estimate; the fit reports what it actually did, and a disagreement
    # between the two is a bug rather than something to average over.
    push!(results, (; stride, batch, nseg = h.nseg, batch_eff = h.batch_eff,
                    upd_per_epoch = h.upd_per_epoch, epochs = h.epochs_run, scored,
                    updates = h.updates, val_every = h.val_every,
                    stopped_early = h.stopped_early, wall = time() - t0,
                    upd_axis = h.update, train = h.train, val = h.val, lrhist = h.lr,
                    best_val = h.best_val, best_update = h.best_update,
                    best_index = h.best_index, ps))
    @printf("    best val %.5g at update %d; final %.5g; %d updates / %d epochs%s; %.1f s\n",
            h.best_val, h.best_update, h.val[end], h.updates, h.epochs_run,
            h.stopped_early ? " (converged, stopped early)" : " (hit the cap)", time() - t0)
    # Saved after EVERY point, atomically. `spec` travels with it, so `LSTMWeights(r.ps, spec)`
    # reconstructs any point's model without re-reading the configuration table.
    m4_save_progress(out; complete = false, results, cell = cfg.name, cfg, seed,
                     epoch_cap = EPOCHS, stop_patience = STOP_PATIENCE, spec,
                     points_done = length(results),
                     points_total = length(POINTS))
    m4_phase("point saved ($(length(results))/$(length(POINTS))) -> $(basename(out))")
end

best_overall = minimum(r.best_val for r in results if isfinite(r.best_val))
THRESHOLDS = (10 * best_overall, 3 * best_overall, 1.5 * best_overall)
# 🔑 Returns the UPDATE the threshold was first met at, read off the history's own update
# axis. The row index is not the update count -- they are `val_every` apart -- and multiplying a
# row index by `upd_per_epoch`, which is what this did while the history was per-epoch, is now
# simply wrong.
first_below(r, thr) = (i = findfirst(<=(thr), r.val); i === nothing ? 0 : r.upd_axis[i])

println()
@printf("%-7s %-6s %5s %5s %5s %7s %7s %5s %11s %9s | %s\n", "stride", "batch", "segs", "beff",
        "u/ep", "epochs", "updates", "conv", "best val", "best upd",
        "updates to reach " * join((@sprintf("%.3g", t) for t in THRESHOLDS), " / "))
println("-"^140)
for r in results
    reach = join((@sprintf("%8d", first_below(r, t)) for t in THRESHOLDS), " ")
    @printf("%-7d %-6d %5d %5d %5d %7d %7d %5s %11.5g %9d | %s\n",
            r.stride, r.batch, r.nseg, r.batch_eff, r.upd_per_epoch, r.epochs, r.updates,
            r.stopped_early ? "yes" : "NO", r.best_val, r.best_update, reach)
end
println("\n(0 in a reach column = never reached; `best upd` is the update the returned iterate " *
        "came from)")
# 🔴 A point that hit the cap did NOT converge, and its best val is a truncation rather than
# a floor -- the exact failure the matched-update policy produced. Say so rather than let the
# table be read as a ranking.
let unconverged = [r for r in results if !r.stopped_early]
    if !isempty(unconverged)
        println("\n🔴 $(length(unconverged)) of $(length(results)) point(s) hit the $EPOCHS-epoch " *
                "cap without converging: " *
                join(("stride $(r.stride) b$(r.batch)" for r in unconverged), ", ") *
                ".\n   Their best validation is where the cap fell, not where the fit went. " *
                "Raise RIKFLOW_M4_EPOCHS before reading them as results.")
    end
end

# The final write adds the thresholds and marks the file complete. ⚠️ `thresholds` are derived from
# the best point in the scan, so they only mean anything once every point has run -- which is
# exactly what `complete` records, and why a partial file must not be read as a ranking.
m4_save_progress(out; complete = true, results, cell = cfg.name, cfg, seed, epoch_cap = EPOCHS,
                 stop_patience = STOP_PATIENCE,
                 spec, points_done = length(results), points_total = length(POINTS),
                 thresholds = THRESHOLDS)
println("\nwrote $out (complete)")
