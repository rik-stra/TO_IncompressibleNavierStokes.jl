# Score online runs against the TRAINING period's level statistics (the calibration targets).
#
#     julia --project=lib/RikFlow/training lib/RikFlow/analysis/m4_calib_score.jl fitdir...
#
# Per run: the 20 TU level mean minus the 1-10 TU mean (in the record's sd) and the level sd over
# the training period's, per QoI; summarised as mean |mean error| and mean |log sd ratio| over the
# six QoIs. Only training-period statistics enter -- the reference band (`m4_screen.jl`) is the
# separate, final test.
using JLD2, Statistics, Printf
TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
ref = load(joinpath(@__DIR__, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
qr = ref["q"]; sdq = vec(std(qr; dims = 2))
tm, ts = vec(mean(qr[:, 400:4000]; dims = 2)), vec(std(qr[:, 400:4000]; dims = 2))
f(v) = join((@sprintf("%6.2f", x) for x in v), "")
@printf("%-26s %-3s  %6s %6s | %-37s | %s\n", "fit", "rep", "|dm|", "|lsr|", "mean error per QoI (sd)", "sd / training sd per QoI")
for d in ARGS
    rows = Tuple{Float64,Float64}[]
    for x in sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d\.jld2$", x), readdir(joinpath(TO, d))))
        o = jldopen(joinpath(TO, d, x), "r") do fh; fh["data_online"]; end
        if size(o.q, 2) < 8001
            @printf("%-26s r%s   diverged at step %d\n", basename(d), x[end-5], size(o.q, 2) - 1); continue
        end
        q = o.q[:, 1:8001]
        dm = (vec(mean(q; dims = 2)) .- tm) ./ sdq
        sr = vec(std(q; dims = 2)) ./ ts
        a, b = mean(abs, dm), mean(abs, log.(sr))
        push!(rows, (a, b))
        @printf("%-26s r%s   %6.3f %6.3f | %s | %s\n", basename(d), x[end-5], a, b, f(dm), f(sr))
    end
    isempty(rows) || @printf("%-26s all  %6.3f %6.3f\n", basename(d), mean(first.(rows)), mean(last.(rows)))
    flush(stdout)
end
