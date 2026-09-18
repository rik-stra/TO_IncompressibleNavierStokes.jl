# Do overlapping (shorter-stride) training segments buy anything?
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_stride_scan.jl [cell] [seed]
#
# Environment: `RIKFLOW_QOI_CACHE` (as the training driver), `RIKFLOW_M4_UPDATES` (optimiser-step
# budget per point, default 3000).
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
# 🔑 **So every point runs to the same number of UPDATES, and the `(400, batch = 2)` row is the
# control that separates (a) from (b).** Comparing at equal epochs instead would confound the two
# completely: the shortest stride would simply have taken eight times as many steps.
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
const UPDATES = parse(Int, get(ENV, "RIKFLOW_M4_UPDATES", "3000"))

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

"Training segments and optimiser steps per epoch, for one (stride, batch)."
function geometry(stride, batch)
    segs = RF.segment_indices(view(steps, 1:ntrain); cfg.L, cfg.burn, stride)
    nseg = length(segs)
    # one group per distinct segment length, each partitioned into chunks of `batch`
    lens = Dict{Int,Int}()
    for s in segs
        lens[length(s.rows)] = get(lens, length(s.rows), 0) + 1
    end
    upd = sum(cld(n, batch) for (_, n) in lens)
    return nseg, upd, sum(length(s.score) for s in segs)
end

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

@info "M4 stride scan" cell=cfg.name arch=cfg.arch beta=cfg.beta seed updates=UPDATES L=cfg.L burn=cfg.burn
@info "data" rows=ncol train_rows=ntrain


# 🔴 Device placement. `M4_DEVICE=cpu` (the default) or `cuda`. `m4_device` refuses `cuda` when no
# device is functional rather than falling back to the host, because a silent fallback would report
# a GPU run that took CPU time. ⚠️ **The GPU path has never been run on a GPU** -- it is verified
# only against `JLArrays`, which enforces the same no-scalar-indexing semantics on the host (V54).
# And expect it to be SLOWER at the current geometry: `results_LSTMS.md` §5.
const DEVICE = RF.m4_device(get(ENV, "M4_DEVICE", "cpu"))
@info "device" M4_DEVICE=get(ENV, "M4_DEVICE", "cpu")

results = NamedTuple[]
for (stride, batch) in POINTS
    nseg, upd_per_epoch, scored = geometry(stride, batch)
    epochs = max(1, cld(UPDATES, upd_per_epoch))
    @info "stride $stride, batch $batch" segments=nseg updates_per_epoch=upd_per_epoch epochs scored_rows_per_epoch=scored
    t0 = time()
    _, h = RF.train_stochlstm(spec, Xc, Yc, steps;
                              cfg.L, cfg.burn, stride, epochs, batch, cfg.lr,
                              cfg.beta, cfg.val_frac, seed, verbose = false, device = DEVICE)
    push!(results, (; stride, batch, nseg, upd_per_epoch, epochs, scored,
                    updates = epochs * upd_per_epoch, wall = time() - t0,
                    train = h.train, val = h.val, lrhist = h.lr,
                    best_val = h.best_val, best_epoch = h.best_epoch))
    @printf("    best val %.5g at epoch %d (update %d); final %.5g; %.1f s\n",
            h.best_val, h.best_epoch, h.best_epoch * upd_per_epoch, h.val[end], time() - t0)
end

best_overall = minimum(r.best_val for r in results if isfinite(r.best_val))
THRESHOLDS = (10 * best_overall, 3 * best_overall, 1.5 * best_overall)
first_below(val, thr) = (i = findfirst(<=(thr), val); i === nothing ? 0 : i)

println()
@printf("%-7s %-6s %5s %5s %7s %11s %9s | %s\n", "stride", "batch", "segs", "u/ep", "epochs",
        "best val", "best upd",
        "updates to reach " * join((@sprintf("%.3g", t) for t in THRESHOLDS), " / "))
println("-"^118)
for r in results
    reach = join((@sprintf("%8d", first_below(r.val, t) * r.upd_per_epoch) for t in THRESHOLDS), " ")
    @printf("%-7d %-6d %5d %5d %7d %11.5g %9d | %s\n",
            r.stride, r.batch, r.nseg, r.upd_per_epoch, r.epochs, r.best_val,
            r.best_epoch * r.upd_per_epoch, reach)
end
println("\n(0 in a reach column = never reached inside the budget; `best upd` is the update the " *
        "returned iterate came from)")

out = joinpath(TO_folder, "stride_scan_$(cfg.name)_seed$(seed).jld2")
jldsave(out; results, cell = cfg.name, cfg, seed, updates = UPDATES, thresholds = THRESHOLDS)
println("\nwrote $out")
