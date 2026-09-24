# Extract what an online run needs from the 2.7 GB tracking record into a small file.
#
#     julia --startup-file=no --project=lib/RikFlow lib/RikFlow/exp_square_HIT/tools/m4_extract_ic.jl
#
# 🔑 `12_online_StochLSTM.jl` loads the whole record (`data_track` is one JLD2 dataset) to use three
# things from it: `params_track`, the first stored field, and the first `nwarm` columns of `dQ`.
# On a 16 GB workstation that is one online run at a time. This writes those three -- ~6 MB -- to
# `output/online_ic_<record>.jld2`, and the driver reads it instead when `RIKFLOW_ONLINE_IC` points at
# it. The contents are copies, so a run from the extract is the same run.

# 🔴 RikFlow, INS and CUDA MUST be loaded: `params_track` holds a `CUDABackend` and RikFlow's
# `FaceAverage`, and without their types in scope JLD2 returns stand-in `Reconstructed*` objects --
# which this would then SAVE, handing the driver a params tuple the solver cannot use.
using JLD2, RikFlow, IncompressibleNavierStokes, CUDA

track_file = get(ENV, "RIKFLOW_TRACK_FILE",
                 joinpath(@__DIR__, "..", "output",
                          "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
out = joinpath(dirname(track_file), "online_ic_" * basename(track_file))
params_track = load(track_file, "params_track")
data_track = load(track_file, "data_track")
u0 = data_track.fields[1].u
# ⚠️ On a machine whose CUDA.jl differs from the writer's, `CUDABackend` lives elsewhere and
# `params_track` comes back as a `JLD2.ReconstructedMutable` whatever is loaded. The driver
# overrides `backend` and `ArrayType` anyway, so rebuild a plain NamedTuple WITHOUT those two and
# fail if any other field is still a stand-in.
if !(params_track isa NamedTuple)
    names = typeof(params_track).parameters[2]
    params_track = (; (k => getproperty(params_track, k) for k in names if !(k in (:backend, :ArrayType)))...)
else
    params_track = Base.structdiff(params_track, NamedTuple{(:backend, :ArrayType)})
end
for (k, v) in pairs(params_track)
    occursin("Reconstructed", string(typeof(v))) && error("params_track.$k is still a JLD2 stand-in: $(typeof(v))")
end
jldsave(out; params_track, u0, dQ = data_track.dQ[:, 1:1000], source = basename(track_file))
println("wrote $out ($(round(filesize(out) / 2^20; digits = 1)) MiB)")
