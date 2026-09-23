# Export one stride-scan point as a deployable M4 fit, so `12_online_StochLSTM.jl` can run it.
#
#     julia --startup-file=no --project=lib/RikFlow/training \
#           lib/RikFlow/exp_square_HIT/tools/m4_export_point.jl <scan_file> <stride> <batch>
#
# e.g. `... m4_export_point.jl stride_scan_StochLSTM2_seed1.jld2 100 32`. `<scan_file>` is a name
# under `output/TO_LSTM/` or a path. Writes
#
#     output/TO_LSTM/<cell>_s<stride>b<batch>_<scan tag>/StochLSTM_seed<seed>.jld2
#
# and prints the `RIKFLOW_M4_MODEL_DIR` to deploy it with.
#
# 🔴 **A scan point is not a deployable fit on its own.** The scan keeps `ps` and `spec` but not
# the scaling, and the online driver finds its model through the configuration table, at
# `output/TO_LSTM/<cfg.name>/` -- which for cell 2 already holds the PRE-split `StochLSTM2` fit.
# Exporting there would overwrite it. So this writes to its OWN directory, and the online driver
# takes that directory through `RIKFLOW_M4_MODEL_DIR`.
#
# 🔑 **The scaling is reconstructed exactly as training computed it**, not stored by the scan:
# `_normalise` of `q[:, a:(b - 1)]` over the cell's `train_range`, used for both input and output --
# the same two lines as `11_train_StochLSTM.jl` and `m4_stride_scan.jl`. Needs the QoI cache (or the
# tracking record), like every M4 driver.
#
# ⚠️ Needs the TRAINING project: `LSTMWeights(ps, spec)`, the precision-to-correlation conversion,
# lives in the Lux extension.

using RikFlow
using Lux, Optimisers, Zygote
using JLD2
using Printf

const RF = RikFlow

length(ARGS) >= 3 || error("usage: julia m4_export_point.jl <scan_file> <stride> <batch>")
TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
scan_path = isfile(ARGS[1]) ? ARGS[1] : joinpath(TO_folder, ARGS[1])
stride, batch = parse(Int, ARGS[2]), parse(Int, ARGS[3])
isfile(scan_path) || error("no scan file at $scan_path")

Base.get_extension(RikFlow, :RikFlowLuxExt) === nothing &&
    error("the Lux extension is not loaded -- run under --project=lib/RikFlow/training")

include(joinpath(@__DIR__, "m4_data.jl"))

d = load(scan_path)
get(d, "complete", false) || @warn "the scan file is not marked complete -- exporting anyway"
matches = [r for r in d["results"] if r.stride == stride && r.batch == batch]
length(matches) == 1 || error("expected exactly one point with stride $stride, batch $batch in " *
                              "$(basename(scan_path)); found $(length(matches)). Points: " *
                              join(("($(r.stride), b$(r.batch))" for r in d["results"]), ", "))
r = only(matches)
cfg, spec, seed = d["cfg"], d["spec"], d["seed"]

# --- the scaling, exactly as training built it ------------------------------------------------
track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(TO_folder, "..",
                          "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
rec = load_m4_qois(; track_file)
a, b = cfg.train_range
_, in_scaling = RF._normalise(rec.q[:, a:(b - 1)]; normalization = cfg.normalization)
scaling = (in_scaling = in_scaling, out_scaling = in_scaling)
all(isfinite, in_scaling.mu) && all(isfinite, in_scaling.sigma) && all(>(0), in_scaling.sigma) ||
    error("reconstructed scaling is not finite/positive: $in_scaling")

w = RF.LSTMWeights(r.ps, spec)
RF.check_shapes(w, spec)

scan_tag = replace(basename(scan_path), "stride_scan_$(cfg.name)_seed$(seed)" => "", ".jld2" => "")
out_dir = joinpath(TO_folder, "$(cfg.name)_s$(stride)b$(batch)" * (isempty(scan_tag) ? "" : scan_tag))
path = joinpath(out_dir, "StochLSTM_seed$(seed).jld2")
isfile(path) && @warn "overwriting an earlier export" path

# `losses` in the shape the training driver writes, so `plot_lstm_losses.jl` and friends read it.
losses = (; update = r.upd_axis, train = r.train, val = r.val, lr = r.lrhist,
          best_val = r.best_val, best_update = r.best_update, best_index = r.best_index,
          stopped_early = r.stopped_early, updates = r.updates, nseg = r.nseg,
          batch_eff = r.batch_eff, upd_per_epoch = r.upd_per_epoch, val_every = r.val_every)
RF.save_stochlstm(path, spec, w, scaling;
                  cfg, seed, train_range = cfg.train_range, track_file, qoi_source = rec.source,
                  losses, ps = r.ps, stride, batch, source_scan = basename(scan_path),
                  stop_reason = get(r, :stop_reason, missing))

# Round-trip through the loader the online driver uses, so a bad export fails here, not on step 1
# of a GPU job.
fit = RF.load_stochlstm(path)
fit.spec.arch == spec.arch && fit.spec.hist.h == spec.hist.h && fit.spec.emission == spec.emission ||
    error("round-trip changed the spec")

@printf("exported %s (stride %d, batch %d): best val %.4g at update %d of %d, %s\n",
        cfg.name, stride, batch, r.best_val, r.best_update, r.updates,
        r.stopped_early ? "stopped early ($(get(r, :stop_reason, "?")))" : "hit the cap")
println("wrote $path")
println("deploy with:  RIKFLOW_M4_MODEL_DIR=$(out_dir) sbatch batch_scripts/run_online.sh lstm " *
        "$(findfirst(c -> c.name == cfg.name, load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")))")
